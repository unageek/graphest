type Ok<T> = {
  ok: T;
  err: undefined;
};

type Err<E> = {
  ok: undefined;
  err: E;
};

export type Result<T, E> = Ok<T> | Err<E>;

export function ok<T>(ok: T): Ok<T> {
  return { ok, err: undefined };
}

export function err<E>(err: E): Err<E> {
  return { ok: undefined, err };
}
