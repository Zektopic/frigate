import { describe, it, expect, vi } from "vitest";
import { validateDetectDimensions } from "../detect";
import type { FormValidation } from "@rjsf/utils";
import type { TFunction } from "i18next";

describe("validateDetectDimensions", () => {
  const mockT: TFunction = vi.fn((key: string) => {
    if (key === "detect.dimensionMustBeEven") {
      return "Dimension must be an even number.";
    }
    return key;
  }) as unknown as TFunction;

  it("should return unmodified errors if formData is not a JsonObject", () => {
    const formData = null;
    const errors: FormValidation = {} as FormValidation;

    const result = validateDetectDimensions(formData, errors, mockT);

    expect(result).toBe(errors);
  });

  it("should return unmodified errors if formData is an Array", () => {
    const formData = [1, 2, 3];
    const errors: FormValidation = {} as FormValidation;

    const result = validateDetectDimensions(formData, errors, mockT);

    expect(result).toBe(errors);
  });

  it("should return unmodified errors if width and height are even", () => {
    const formData = { width: 640, height: 480 };
    const widthAddError = vi.fn();
    const heightAddError = vi.fn();
    const errors: FormValidation = {
      width: { addError: widthAddError },
      height: { addError: heightAddError },
    } as unknown as FormValidation;

    const result = validateDetectDimensions(formData, errors, mockT);

    expect(widthAddError).not.toHaveBeenCalled();
    expect(heightAddError).not.toHaveBeenCalled();
    expect(result).toBe(errors);
  });

  it("should add error for odd width", () => {
    const formData = { width: 641, height: 480 };
    const widthAddError = vi.fn();
    const heightAddError = vi.fn();
    const errors: FormValidation = {
      width: { addError: widthAddError },
      height: { addError: heightAddError },
    } as unknown as FormValidation;

    validateDetectDimensions(formData, errors, mockT);

    expect(widthAddError).toHaveBeenCalledWith("Dimension must be an even number.");
    expect(heightAddError).not.toHaveBeenCalled();
  });

  it("should add error for odd height", () => {
    const formData = { width: 640, height: 481 };
    const widthAddError = vi.fn();
    const heightAddError = vi.fn();
    const errors: FormValidation = {
      width: { addError: widthAddError },
      height: { addError: heightAddError },
    } as unknown as FormValidation;

    validateDetectDimensions(formData, errors, mockT);

    expect(widthAddError).not.toHaveBeenCalled();
    expect(heightAddError).toHaveBeenCalledWith("Dimension must be an even number.");
  });

  it("should add errors for both odd width and odd height", () => {
    const formData = { width: 641, height: 481 };
    const widthAddError = vi.fn();
    const heightAddError = vi.fn();
    const errors: FormValidation = {
      width: { addError: widthAddError },
      height: { addError: heightAddError },
    } as unknown as FormValidation;

    validateDetectDimensions(formData, errors, mockT);

    expect(widthAddError).toHaveBeenCalledWith("Dimension must be an even number.");
    expect(heightAddError).toHaveBeenCalledWith("Dimension must be an even number.");
  });

  it("should ignore missing width and height", () => {
    const formData = {};
    const widthAddError = vi.fn();
    const heightAddError = vi.fn();
    const errors: FormValidation = {
      width: { addError: widthAddError },
      height: { addError: heightAddError },
    } as unknown as FormValidation;

    validateDetectDimensions(formData, errors, mockT);

    expect(widthAddError).not.toHaveBeenCalled();
    expect(heightAddError).not.toHaveBeenCalled();
  });

  it("should ignore non-number width and height", () => {
    const formData = { width: "640", height: "480" };
    const widthAddError = vi.fn();
    const heightAddError = vi.fn();
    const errors: FormValidation = {
      width: { addError: widthAddError },
      height: { addError: heightAddError },
    } as unknown as FormValidation;

    validateDetectDimensions(formData, errors, mockT);

    expect(widthAddError).not.toHaveBeenCalled();
    expect(heightAddError).not.toHaveBeenCalled();
  });

  it("should handle missing addError gracefully", () => {
    const formData = { width: 641, height: 481 };
    const errors: FormValidation = {
      width: {},
      height: {},
    } as unknown as FormValidation;

    // Should not throw
    expect(() => validateDetectDimensions(formData, errors, mockT)).not.toThrow();
  });

  it("should handle missing width/height in errors object", () => {
    const formData = { width: 641, height: 481 };
    const errors: FormValidation = {} as unknown as FormValidation;

    // Should not throw
    expect(() => validateDetectDimensions(formData, errors, mockT)).not.toThrow();
  });
});
