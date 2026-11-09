# Security Policy

## Supported versions

Only the latest release on the [releases page](https://github.com/Yamabiko-ASR/yamabiko/releases)
receives security fixes. Beta versions before it are unsupported.

## Reporting a vulnerability

Please report vulnerabilities **privately**, not via public issues:

- Email: [callen00@proton.me](mailto:callen00@proton.me)
- Or use GitHub's private vulnerability reporting: on the **Security** tab of this repository, click **Report a vulnerability**.

You can expect an acknowledgement within a week. Please include
reproduction steps and the Yamabiko version.

## Scope notes

Yamabiko is a local desktop application with no account service, telemetry
service, or cloud audio/text processing. It makes outbound requests for model
downloads, update checks, and optional AniList lookups. Surfaces most relevant
to security review:

- The localhost WebSocket and HTTP overlay server (per-launch session token, origin policy, and command handling) and the localhost llama.cpp endpoint.
- The NSIS installer and the per-user install layout under
  `%LOCALAPPDATA%\Yamabiko`.
- Subtitle text rendering (transcribed/translated content is
  attacker-influenced when users watch untrusted streams).
- Model downloads from Hugging Face during first-run setup.
- Series lookups against the AniList API (remote JSON parsed by the gazetteer, and cover images fetched, saved next to the series you pick, and shown in the series bar and the tray flyout).
