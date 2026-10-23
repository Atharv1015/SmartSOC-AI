"""
SmartSOC Mitigation Executor v1

Defensive endpoint mitigation layer for the SmartSOC
controlled blue-team research environment.

Supported defensive capabilities:

    PROCESS_TERMINATION
    FILE_QUARANTINE
    ACCOUNT_CONTAINMENT

Design principles:

    - Defensive use only
    - Explicit capability selection
    - Explicit target validation
    - No arbitrary shell commands
    - Verification after mitigation
    - Audit-friendly result structure
    - Reversible operations where practical

This module is intended for the SmartSOC Windows
victim/lab endpoint.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class MitigationExecutor:
    """
    Executes controlled defensive mitigations on the
    designated SmartSOC lab endpoint.
    """

    VERSION = "mitigation-executor-v1"

    ALLOWED_ACTIONS = {
        "PROCESS_TERMINATION",
        "FILE_QUARANTINE",
        "ACCOUNT_CONTAINMENT",
    }

    def __init__(
        self,
        quarantine_directory: Path | None = None,
    ):

        if quarantine_directory is None:

            quarantine_directory = (
                Path.home()
                / "SmartSOC-Quarantine"
            )

        self.quarantine_directory = (
            Path(quarantine_directory)
        )

        self.quarantine_directory.mkdir(
            parents=True,
            exist_ok=True,
        )


    # ------------------------------------------------------------------
    # COMMON HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _timestamp() -> str:

        return datetime.now(
            timezone.utc
        ).isoformat()


    @staticmethod
    def _sha256(
        file_path: Path,
    ) -> str:

        digest = hashlib.sha256()

        with file_path.open(
            "rb"
        ) as handle:

            for chunk in iter(
                lambda: handle.read(1024 * 1024),
                b"",
            ):

                digest.update(chunk)

        return digest.hexdigest()


    @staticmethod
    def _result(
        action: str,
        target: Any,
        success: bool,
        verified: bool,
        message: str,
        **extra: Any,
    ) -> Dict[str, Any]:

        return {
            "executor": MitigationExecutor.VERSION,
            "timestamp": MitigationExecutor._timestamp(),
            "action": action,
            "target": target,
            "success": success,
            "verified": verified,
            "message": message,
            **extra,
        }


    # ------------------------------------------------------------------
    # ACTION VALIDATION
    # ------------------------------------------------------------------

    def validate_action(
        self,
        action: str,
    ) -> None:

        if not isinstance(
            action,
            str,
        ):

            raise TypeError(
                "Mitigation action must be a string."
            )

        normalized = (
            action.strip().upper()
        )

        if normalized not in self.ALLOWED_ACTIONS:

            raise ValueError(
                f"Unsupported mitigation action: "
                f"{action}"
            )


    # ------------------------------------------------------------------
    # PROCESS TERMINATION
    # ------------------------------------------------------------------

    def terminate_process(
        self,
        pid: int,
    ) -> Dict[str, Any]:

        action = "PROCESS_TERMINATION"

        self.validate_action(
            action
        )

        if not isinstance(
            pid,
            int,
        ) or pid <= 0:

            return self._result(
                action=action,
                target=pid,
                success=False,
                verified=False,
                message="Invalid process ID.",
            )

        try:

            process = subprocess.run(
                [
                    "taskkill",
                    "/PID",
                    str(pid),
                    "/T",
                    "/F",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                shell=False,
            )

            verification = subprocess.run(
                [
                    "tasklist",
                    "/FI",
                    f"PID eq {pid}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                shell=False,
            )

            process_still_present = (
                str(pid)
                in verification.stdout
            )

            verified = not process_still_present

            return self._result(
                action=action,
                target={
                    "pid": pid,
                },
                success=(
                    process.returncode == 0
                    and verified
                ),
                verified=verified,
                message=(
                    "Process terminated and "
                    "termination verified."
                    if verified
                    else
                    "Process termination could not "
                    "be verified."
                ),
                return_code=process.returncode,
                stdout=process.stdout,
                stderr=process.stderr,
            )

        except Exception as exc:

            return self._result(
                action=action,
                target={
                    "pid": pid,
                },
                success=False,
                verified=False,
                message=(
                    "Process termination failed."
                ),
                error=str(exc),
            )


    # ------------------------------------------------------------------
    # FILE QUARANTINE
    # ------------------------------------------------------------------

    def quarantine_file(
        self,
        file_path: str,
    ) -> Dict[str, Any]:

        action = "FILE_QUARANTINE"

        self.validate_action(
            action
        )

        source = Path(
            file_path
        ).resolve()

        if not source.exists():

            return self._result(
                action=action,
                target=str(source),
                success=False,
                verified=False,
                message="Target file does not exist.",
            )

        if not source.is_file():

            return self._result(
                action=action,
                target=str(source),
                success=False,
                verified=False,
                message="Target is not a regular file.",
            )

        try:

            original_hash = (
                self._sha256(source)
            )

            timestamp = (
                datetime.now(
                    timezone.utc
                )
                .strftime(
                    "%Y%m%dT%H%M%SZ"
                )
            )

            quarantine_name = (
                f"{timestamp}_"
                f"{original_hash[:16]}_"
                f"{source.name}"
            )

            destination = (
                self.quarantine_directory
                / quarantine_name
            )

            shutil.move(
                str(source),
                str(destination),
            )

            destination_hash = (
                self._sha256(
                    destination
                )
            )

            source_removed = (
                not source.exists()
            )

            hash_preserved = (
                original_hash
                == destination_hash
            )

            verified = (
                source_removed
                and hash_preserved
                and destination.exists()
            )

            return self._result(
                action=action,
                target=str(source),
                success=verified,
                verified=verified,
                message=(
                    "File quarantined and verified."
                    if verified
                    else
                    "File quarantine verification failed."
                ),
                original_path=str(source),
                quarantine_path=str(destination),
                original_sha256=original_hash,
                quarantine_sha256=destination_hash,
            )

        except Exception as exc:

            return self._result(
                action=action,
                target=str(source),
                success=False,
                verified=False,
                message="File quarantine failed.",
                error=str(exc),
            )


    # ------------------------------------------------------------------
    # ACCOUNT CONTAINMENT
    # ------------------------------------------------------------------

    def contain_account(
        self,
        username: str,
    ) -> Dict[str, Any]:

        action = "ACCOUNT_CONTAINMENT"

        self.validate_action(
            action
        )

        if not isinstance(
            username,
            str,
        ):

            return self._result(
                action=action,
                target=username,
                success=False,
                verified=False,
                message="Invalid username.",
            )

        username = username.strip()

        if not username:

            return self._result(
                action=action,
                target=username,
                success=False,
                verified=False,
                message="Username cannot be empty.",
            )

        try:

            disable_result = subprocess.run(
                [
                    "net",
                    "user",
                    username,
                    "/active:no",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                shell=False,
            )

            verification = subprocess.run(
                [
                    "net",
                    "user",
                    username,
                ],
                capture_output=True,
                text=True,
                timeout=15,
                shell=False,
            )

            output = (
                verification.stdout
                + "\n"
                + verification.stderr
            ).lower()

            disabled = (
                "account active"
                in output
                and "no"
                in output
            )

            return self._result(
                action=action,
                target={
                    "username": username,
                },
                success=(
                    disable_result.returncode == 0
                    and disabled
                ),
                verified=disabled,
                message=(
                    "Account containment completed "
                    "and verified."
                    if disabled
                    else
                    "Account containment could not "
                    "be verified."
                ),
                return_code=(
                    disable_result.returncode
                ),
                stdout=disable_result.stdout,
                stderr=disable_result.stderr,
            )

        except Exception as exc:

            return self._result(
                action=action,
                target={
                    "username": username,
                },
                success=False,
                verified=False,
                message="Account containment failed.",
                error=str(exc),
            )


    # ------------------------------------------------------------------
    # GENERIC DEFENSIVE DISPATCHER
    # ------------------------------------------------------------------

    def execute(
        self,
        action: str,
        target: Any,
    ) -> Dict[str, Any]:

        normalized = (
            action.strip().upper()
            if isinstance(action, str)
            else action
        )

        if normalized == "PROCESS_TERMINATION":

            return self.terminate_process(
                int(target)
            )

        if normalized == "FILE_QUARANTINE":

            return self.quarantine_file(
                str(target)
            )

        if normalized == "ACCOUNT_CONTAINMENT":

            return self.contain_account(
                str(target)
            )

        return self._result(
            action=str(action),
            target=target,
            success=False,
            verified=False,
            message="Unsupported mitigation action.",
        )


# ----------------------------------------------------------------------
# LAB TEST
# ----------------------------------------------------------------------

def main() -> None:

    print("=" * 72)
    print("SmartSOC MITIGATION EXECUTOR v1")
    print("=" * 72)

    executor = MitigationExecutor()

    print()
    print("[CAPABILITIES]")

    for action in sorted(
        executor.ALLOWED_ACTIONS
    ):

        print(
            " ",
            action,
        )

    print()
    print(
        "Executor version:",
        executor.VERSION,
    )

    print()
    print(
        "No mitigation is executed by this self-test."
    )


if __name__ == "__main__":
    main()