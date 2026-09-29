import { useState, useEffect, useRef } from "react";
import axios from "axios";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import ActivityIndicator from "@/components/indicators/activity-indicator";
import { baseUrl } from "@/api/baseUrl";

import { useTranslation } from "react-i18next";

// How often to probe the backend while it restarts.
const RESTART_POLL_MS = 2000;

type RestartDialogProps = {
  isOpen: boolean;
  onClose: () => void;
  onRestart: () => void;
};

export default function RestartDialog({
  isOpen,
  onClose,
  onRestart,
}: RestartDialogProps) {
  const { t } = useTranslation("components/dialog");
  const [restartDialogOpen, setRestartDialogOpen] = useState(isOpen);
  const [restartingSheetOpen, setRestartingSheetOpen] = useState(false);
  const [countdown, setCountdown] = useState(60);
  const countdownRef = useRef(countdown);

  useEffect(() => {
    setRestartDialogOpen(isOpen);
  }, [isOpen]);

  useEffect(() => {
    countdownRef.current = countdown;
  }, [countdown]);

  useEffect(() => {
    let countdownInterval: NodeJS.Timeout;

    if (restartingSheetOpen) {
      countdownInterval = setInterval(() => {
        setCountdown((prevCountdown) => Math.max(prevCountdown - 1, 0));
      }, 1000);
    }

    return () => {
      clearInterval(countdownInterval);
    };
  }, [restartingSheetOpen]);

  // Reload once the backend is answering again rather than after a fixed
  // 60s: a slow restart used to reload onto a dead server (the browser's
  // own error page, or an app whose first requests all failed and then sat
  // in SWR's error backoff), which is what left users reloading by hand.
  useEffect(() => {
    if (!restartingSheetOpen) {
      return;
    }

    let sawBackendDown = false;
    const pollInterval = setInterval(async () => {
      try {
        await axios.get("version", { timeout: RESTART_POLL_MS });
      } catch {
        sawBackendDown = true;
        return;
      }

      if (sawBackendDown || countdownRef.current <= 0) {
        window.location.href = baseUrl;
      }
    }, RESTART_POLL_MS);

    return () => {
      clearInterval(pollInterval);
    };
  }, [restartingSheetOpen]);

  const handleRestart = () => {
    setRestartingSheetOpen(true);
    onRestart();
  };

  const handleForceReload = () => {
    window.location.href = baseUrl;
  };

  return (
    <>
      <AlertDialog
        open={restartDialogOpen}
        onOpenChange={(open) => {
          if (!open) {
            setRestartDialogOpen(false);
            onClose();
          }
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("restart.title")}</AlertDialogTitle>
            <AlertDialogDescription className="sr-only">
              {t("restart.description")}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>
              {t("button.cancel", { ns: "common" })}
            </AlertDialogCancel>
            <AlertDialogAction onClick={handleRestart}>
              {t("restart.button")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <Sheet
        open={restartingSheetOpen}
        onOpenChange={() => setRestartingSheetOpen(false)}
      >
        <SheetContent
          side="top"
          onInteractOutside={(e) => e.preventDefault()}
          className="[&>button:first-of-type]:hidden"
        >
          <div className="flex flex-col items-center">
            <ActivityIndicator />
            <SheetHeader className="mt-5 text-center">
              <SheetTitle className="text-center">
                {t("restart.restarting.title")}
              </SheetTitle>
              <SheetDescription className="text-center">
                <div>
                  {countdown > 0
                    ? t("restart.restarting.content", {
                        countdown,
                      })
                    : t("restart.restarting.waiting")}
                </div>
              </SheetDescription>
            </SheetHeader>
            <Button
              size="lg"
              className="mt-5"
              aria-label={t("restart.restarting.button")}
              onClick={handleForceReload}
            >
              {t("restart.restarting.button")}
            </Button>
          </div>
        </SheetContent>
      </Sheet>
    </>
  );
}
