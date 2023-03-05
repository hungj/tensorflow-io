# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"Command line tool to analyze benchmark result with Welch's t-test."

import argparse
import collections
import logging
import json
import os
from operator import attrgetter

import numpy as np
from scipy.stats import t

# Common field names used in pytest-benchmark JSON report.
BENCHMARKS = "benchmarks"
GROUP = "group"
NAME = "name"
STATS = "stats"
MEAN = "mean"
STDDEV = "stddev"
ROUNDS = "rounds"


class BenchmarkName(
    collections.namedtuple(
        "BenchmarkName",
        ["group", "name"])):
  """The name and group of a pytest-benchmark test.

  Fields:
    group: The name of the pytest-benchmark group that this test belongs to.
           A benchmark group contains one or more benchmark tests.
    name: The name of the benchmark test.
  """
  pass


class BenchmarkResult(
    collections.namedtuple(
        "BenchmarkResult",
        ["label", "result"])):
  """Benchmark result loaded from pytest-benchmark JSON report.

  Fields:
    label: The label of the benchmark run.
    result: A benchmark dict generated by the pytest-benchmark JSON report.
            Key is a BenchmarkName. Value is a stat dict. Stat dict contains
            statistics such as min, max, mean, std, etc
  """
  pass


class TTestResult(
    collections.namedtuple(
        "TTestResult",
        ["p_value", "t_stat", "lower_bound", "upper_bound", "mean_delta"])):
  """Welch's t-test result and one sided confidence interval.

  Fields:
    p_value: The p value of the Welch's t-test.
    t_stat: The T statistic of the Welch's t-test.
    lower_bound: The lower bound of the one sided confidence interval.
    upper_bound: The upper bound of the one sided confidence interval.
    mean_delta: The difference in sample mean.
  """
  pass


class AlphaAction(argparse.Action):
  """Parser action for confidence level alpha to validate its value."""

  def __call__(self, parser, namespace, values, option_string=None):
    if values <= 0.0 or values >= 1.0:
      parser.error(
        f"The alpha range of '{option_string}' should be (0.0, 1.0)."
      )

    setattr(namespace, self.dest, values)


def load_benchmark_result(path):
  filename = os.path.basename(path)
  label = os.path.splitext(filename)[0]  # Remove file extension

  with open(path, 'r') as f:
    report = json.load(f)

  result = {}
  for benchmark_test in report[BENCHMARKS]:
    group_name = benchmark_test[GROUP]
    test_name = benchmark_test[NAME]

    benchmark_name = BenchmarkName(group=group_name, name=test_name)
    result[benchmark_name] = benchmark_test[STATS]

  return BenchmarkResult(label=label, result=result)


def run_welchs_ttest(stat1, stat2, alpha, faster):
  """Run one tailed Welch's t-test to verify if stat1 is faster/slower than stat2

  Please refer wiki for more details about Welch's t-test.
  https://en.wikipedia.org/wiki/Welch%27s_t-test

  Please check scipy for the t-test implementation details.
  https://github.com/scipy/scipy/blob/v1.7.1/scipy/stats/stats.py#L5712-L5833

  The confidence interval is computed with one sided approach. For more details, see
  https://stats.stackexchange.com/questions/257526/can-one-sided-confidence-intervals-have-95-coverage

  Args:
    stat1: The first statistic dict collected by pytest-benchmark.
    stat2: The second statistic dict collected by pytest-benchmark.
    alpha: The confidence level.
    faster: True to test if stat1 is faster than stat2. False to test
            if stat1 is slower than stat2.

  Returns:
    A TTestResult
  """
  m1 = stat1[MEAN]
  m2 = stat2[MEAN]

  s1 = stat1[STDDEV]
  s2 = stat2[STDDEV]

  n1 = stat1[ROUNDS]
  n2 = stat2[ROUNDS]

  df1 = n1 - 1  # degree of freedom of stat1
  df2 = n2 - 1  # degree of freedom of stat2

  sample_v1 = s1**2 / n1  # biased estimated sample variance of stat1
  sample_v2 = s2**2 / n2  # biased estimated sample variance of stat2

  biased_variance = np.sqrt(sample_v1 + sample_v2)
  # degree of freedom
  df = (sample_v1 + sample_v2)**2 / (sample_v1**2/(df1) + sample_v2**2/(df2))

  mean_delta = m1 - m2
  t_stat = mean_delta / biased_variance

  if faster:
    # Null hypothesis is stat1 >= stat2.
    # Alternative hypothesis is stat1 < stat2.
    p_value = t.cdf(t_stat, df)

    # Compute one sided confidence interval (-inf, x)
    upper_bound = mean_delta + t.ppf(1.0 - alpha, df) * biased_variance
    upper_bound = format(upper_bound, '.5f')
    lower_bound = "-inf"
  else:
    # Null hypothesis is stat1 <= stat2.
    # Alternative hypothesis is stat1 > stat2.
    p_value = 1.0 - t.cdf(t_stat, df)

    # Compute one sided confidence interval (x, inf)
    upper_bound = "inf"
    lower_bound = mean_delta + t.ppf(alpha, df) * biased_variance
    lower_bound = format(lower_bound, '.5f')

  return TTestResult(p_value=p_value, t_stat=t_stat,
                     lower_bound=lower_bound, upper_bound=upper_bound,
                     mean_delta=format(mean_delta, '.5f'))


def create_result_messages(benchmark_results, alpha):
  sorted_benchmarks = sorted(list(benchmark_results.keys()),
                             key=attrgetter("group", "name"))

  template = "\t{benchmark:55}{mean_delta:25}{confidence_interval:40}"

  confidence = (1.0 - alpha) * 100
  results = [
    template.format(
      benchmark="Benchmark test (group::name)",
      mean_delta="Mean delta in second",
      confidence_interval=f"{confidence}% confidence interval of mean delta"
    )
  ]
  for benchmark in sorted_benchmarks:
    ttest_result = benchmark_results[benchmark]
    results.append(
      template.format(
        benchmark=f"{benchmark.group}::{benchmark.name}",
        mean_delta=f"{ttest_result.mean_delta}",
        confidence_interval=f"({ttest_result.lower_bound}, "  \
                            f"{ttest_result.upper_bound})"
      )
    )

  return '\n'.join(results)


def log_benchmark_not_in_both_report(diff, in_result, not_in_result):
  sorted_diff = sorted(list(diff), key=attrgetter('group', 'name'))
  sorted_benchmarks = '\n'.join([
    f"\t{benchmark.group}::{benchmark.name}" for benchmark in sorted_diff
  ])

  message = f"Found following benchmarks in {in_result.label} "  \
            f"but not in {not_in_result.label}.\n {sorted_benchmarks}"
  logging.warning(message)


def log_no_overlapped_benchmark_result(first_result, second_result):
  message = f"Benchmark results in {first_result.label} "  \
            f"and {second_result.label} have no intersection."
  logging.warning(message)


def log_not_significant_benchmark_result(benchmarks, label1, label2,
                                         order, alpha):
  result_messages = create_result_messages(benchmarks, alpha)
  message = f"The following benchmark results does NOT show that "  \
            f"{label1} is statistically significant {order} than "  \
            f"{label2}.\n {result_messages}"
  logging.info(message)


def log_significant_benchmark_result(benchmarks, label1, label2,
                                     order, alpha):
  result_messages = create_result_messages(benchmarks, alpha)
  message = f"The following benchmark results show that {label1} is "  \
            f"statistically significant {order} than {label2}."  \
            f"\n{result_messages}"
  logging.info(message)


def parse_args():
  parser = argparse.ArgumentParser(
    description="Run Welch's t-test on benchmark result.")
  parser.add_argument('first_report', metavar='r1', type=str,
                      help='Path to the first pytest-benchmark JSON report.')
  parser.add_argument('second_report', metavar='r2', type=str,
                      help='Path to the second pytest-benchmark JSON report.')
  parser.add_argument('-a', '--alpha', metavar='alpha', type=float, default=0.05,
                      action=AlphaAction, help='The confidence level in t-test.')

  slower_message = "Set this flag to test if the first result is slower "  \
                   "than the second. Otherwise, the tool will test if the "  \
                   "first result is faster than the second."
  parser.add_argument('--slower', action='store_true', help=slower_message)

  return parser.parse_args()


def main():
  args = parse_args()

  first_result = load_benchmark_result(args.first_report)
  second_result = load_benchmark_result(args.second_report)

  benchmark_in_first = set(first_result.result.keys())
  benchmark_in_second = set(second_result.result.keys())

  first_but_not_second = benchmark_in_first.difference(benchmark_in_second)
  second_but_not_first = benchmark_in_second.difference(benchmark_in_first)
  in_both = benchmark_in_first.intersection(benchmark_in_second)

  if first_but_not_second:
    log_benchmark_not_in_both_report(
      diff=first_but_not_second,
      in_result=first_result,
      not_in_result=second_result)

  if second_but_not_first:
    log_benchmark_not_in_both_report(
      diff=second_but_not_first,
      in_result=second_result,
      not_in_result=first_result)

  if not in_both:
    log_no_overlapped_benchmark_result(first_result, second_result)
    return

  is_faster = False if args.slower else True
  alpha = args.alpha

  significant = {}
  not_significant = {}
  for benchmark_test in in_both:
    ttest_result = run_welchs_ttest(
      stat1=first_result.result[benchmark_test],
      stat2=second_result.result[benchmark_test],
      alpha=alpha,
      faster=is_faster)

    if ttest_result.p_value < alpha:
      significant[benchmark_test] = ttest_result
    else:
      not_significant[benchmark_test] = ttest_result

  order = "faster" if is_faster else "slower"

  if not_significant:
    log_not_significant_benchmark_result(not_significant, first_result.label,
                                         second_result.label, order, alpha)

  if significant:
    log_significant_benchmark_result(significant, first_result.label,
                                     second_result.label, order, alpha)


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  main()
