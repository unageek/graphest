# Regression Test Suite

## Notes

- Assign each test with a UUID (v4) and prefix it with `t_` to make an identifier.

- Run tests with:

  ```bash
  cargo t --release --test graph
  ```

- You can time each test with:

  ```bash
  cargo t --release --test graph -- -Zunstable-options --report-time
  ```
