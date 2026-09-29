import { useEffect } from "react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { reloadIfNotRecentlyReloaded } from "@/utils/chunkLoadError";

// How often to check whether Frigate is reachable again after a route chunk
// failed to load.
const REACHABLE_POLL_MS = 3000;

type ErrorFallbackProps = {
  chunkError: boolean;
};

export default function ErrorFallback({ chunkError }: ErrorFallbackProps) {
  const { t } = useTranslation(["common"]);

  useEffect(() => {
    if (!chunkError) {
      return;
    }

    // Reload only once the server answers again: reloading while it is
    // still down would replace the app with the browser's own error page.
    const interval = setInterval(async () => {
      try {
        const response = await fetch(window.location.href, {
          method: "HEAD",
          cache: "no-store",
        });

        if (response.ok) {
          reloadIfNotRecentlyReloaded();
        }
      } catch {
        // still unreachable, try again on the next tick
      }
    }, REACHABLE_POLL_MS);

    return () => clearInterval(interval);
  }, [chunkError]);

  return (
    <div className="flex size-full flex-col items-center justify-center gap-4 p-4 text-center">
      <div className="text-lg">{t("errorBoundary.title")}</div>
      <div className="max-w-md text-sm text-muted-foreground">
        {chunkError
          ? t("errorBoundary.chunkDescription")
          : t("errorBoundary.description")}
      </div>
      <Button
        aria-label={t("errorBoundary.reload")}
        onClick={() => window.location.reload()}
      >
        {t("errorBoundary.reload")}
      </Button>
    </div>
  );
}
