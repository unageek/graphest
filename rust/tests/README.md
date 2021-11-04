# Regression Test Suite

## Notes

- The name of each test case must be a UUID (variant 1, version 4) prefixed with `t_`.

- Run tests with:

  ```bash
  cargo t --release --test graph -- --test-threads=4
  ```

  Each execution can consume up to ~1 GiB of RAM (peak usage can be higher), so here we use `--test-threads=4` to limit the number of concurrent executions.

- You can time each test with:

  ```bash
  cargo t --release --test graph -- --test-threads=4 -Zunstable-options --report-time
  ```
