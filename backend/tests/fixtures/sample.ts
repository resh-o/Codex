import { strict as assert } from "assert";

/** Adds two numbers together. */
export function add(a: number, b: number): number {
  function inner(n: number): number {
    return n * 2;
  }
  return inner(a) + b;
}

// A simple user authentication class
export class UserAuth {
  private token: string;

  constructor(token: string) {
    this.token = token;
  }

  /** Validate a candidate token. */
  validateToken(candidate: string): boolean {
    return candidate === this.token;
  }
}

const multiply = (a: number, b: number): number => a * b;
