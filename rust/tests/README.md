# Regression Test Suite

## Notes

- Assign each test with a UUID (variant 1, version 4) and prefix it with `t_` to make an identifier.

- Run tests with:

  ```bash
  cargo t --release --test graph -- --test-threads=4
  ```

  Each execution can consume up to ~1GiB of RAM (peak usage can be higher), so here we use `--test-threads=4` to limit the number of concurrent executions.

- You can time each test with:

  ```bash
  cargo t --release --test graph -- --test-threads=4 -Zunstable-options --report-time
  ```
