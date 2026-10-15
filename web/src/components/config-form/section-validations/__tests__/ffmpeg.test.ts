import { describe, it, expect, vi, beforeEach } from "vitest";
import { validateFfmpegInputRoles } from "../ffmpeg";
import type { FormValidation } from "@rjsf/utils";
import type { TFunction } from "i18next";

describe("validateFfmpegInputRoles", () => {
  let mockErrors: FormValidation;
  let mockT: TFunction;
  let mockAddError: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockAddError = vi.fn();
    mockErrors = {
      inputs: {
        addError: mockAddError,
      },
    } as unknown as FormValidation;

    mockT = vi.fn().mockImplementation((key) => key) as unknown as TFunction;
  });

  it("returns errors unmodified if formData is not an object", () => {
    const result = validateFfmpegInputRoles("not an object", mockErrors, mockT);
    expect(result).toBe(mockErrors);
    expect(mockAddError).not.toHaveBeenCalled();
  });

  it("returns errors unmodified if formData.inputs is not an array", () => {
    const result = validateFfmpegInputRoles(
      { inputs: "not an array" },
      mockErrors,
      mockT,
    );
    expect(result).toBe(mockErrors);
    expect(mockAddError).not.toHaveBeenCalled();
  });

  it("validates successfully with detect role and no duplicates", () => {
    const formData = {
      inputs: [{ roles: ["detect"] }, { roles: ["record"] }],
    };
    const result = validateFfmpegInputRoles(formData, mockErrors, mockT);
    expect(result).toBe(mockErrors);
    expect(mockAddError).not.toHaveBeenCalled();
  });

  it("adds error when detect role is missing", () => {
    const formData = {
      inputs: [{ roles: ["record"] }, { roles: ["audio"] }],
    };
    validateFfmpegInputRoles(formData, mockErrors, mockT);
    expect(mockAddError).toHaveBeenCalledWith("ffmpeg.inputs.detectRequired");
  });

  it("adds error when duplicate roles exist across inputs", () => {
    const formData = {
      inputs: [{ roles: ["detect", "record"] }, { roles: ["record"] }],
    };
    validateFfmpegInputRoles(formData, mockErrors, mockT);
    expect(mockAddError).toHaveBeenCalledWith("ffmpeg.inputs.rolesUnique");
    // Also missing detect check shouldn't trigger since 'detect' is present
    expect(mockAddError).not.toHaveBeenCalledWith(
      "ffmpeg.inputs.detectRequired",
    );
  });

  it("adds error when hwaccel_args is present on input without detect role", () => {
    const formData = {
      inputs: [
        { roles: ["detect"] },
        { roles: ["record"], hwaccel_args: "preset-vaapi" },
      ],
    };
    validateFfmpegInputRoles(formData, mockErrors, mockT);
    expect(mockAddError).toHaveBeenCalledWith(
      "ffmpeg.inputs.hwaccelDetectOnly",
    );
  });

  it("handles empty arrays/null for hwaccel_args gracefully", () => {
    const formData = {
      inputs: [
        { roles: ["detect"] },
        { roles: ["record"], hwaccel_args: [] },
        { roles: ["rtmp"], hwaccel_args: null },
      ],
    };
    validateFfmpegInputRoles(formData, mockErrors, mockT);
    expect(mockAddError).not.toHaveBeenCalled();
  });

  it("allows hwaccel_args when detect role is present", () => {
    const formData = {
      inputs: [{ roles: ["detect"], hwaccel_args: "preset-vaapi" }],
    };
    validateFfmpegInputRoles(formData, mockErrors, mockT);
    expect(mockAddError).not.toHaveBeenCalled();
  });
  it("ignores inputs that are not objects or have missing/invalid roles array", () => {
    const formData = {
      inputs: [
        "not an object",
        { noRolesArray: true },
        { roles: "not an array" },
        { roles: ["detect"] },
      ],
    };
    validateFfmpegInputRoles(formData, mockErrors, mockT);
    expect(mockAddError).not.toHaveBeenCalled();
  });

  it("ignores roles that are not strings", () => {
    const formData = {
      inputs: [
        { roles: ["detect", 123, null, undefined, {}] }
      ],
    };
    validateFfmpegInputRoles(formData, mockErrors, mockT);
    expect(mockAddError).not.toHaveBeenCalled();
  });
});
