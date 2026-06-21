import { describe, test, expect, beforeEach } from "vitest";
import { isRedirectingToLogin, setRedirectingToLogin } from "../auth-redirect";

describe("auth-redirect", () => {
  beforeEach(() => {
    // Reset state before each test
    setRedirectingToLogin(false);
  });

  test("should initially not be redirecting", () => {
    expect(isRedirectingToLogin()).toBe(false);
  });

  test("should update state when setRedirectingToLogin is called", () => {
    setRedirectingToLogin(true);
    expect(isRedirectingToLogin()).toBe(true);

    setRedirectingToLogin(false);
    expect(isRedirectingToLogin()).toBe(false);
  });
});
