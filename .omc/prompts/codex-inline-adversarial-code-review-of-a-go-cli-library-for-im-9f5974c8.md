Adversarial code review of a Go CLI + library for image generation via Replicate and OpenAI Image API.

Goals: identify REAL defects, security issues, correctness bugs, race conditions, resource leaks, API misuse, missing-edge-case handling, test gaps, or subtle behavior issues. Skip stylistic nits unless they cause bugs. Be ruthless and concrete — file:line references, why it's wrong, and what to do.

Specific things to scrutinize:
1. HTTP request lifecycle: are response bodies always drained/closed? Any leaks on error paths?
2. The multipart/form-data goroutine in OpenAI edits — proper close ordering? deadlock potential? error propagation through io.Pipe?
3. Replicate "Prefer: wait=60" semantics: if the prediction succeeds within the wait, do we still try to poll? Could pred.URLs["get"] be empty, causing a noisy error?
4. Token-in-URL detection in Replicate downloads — false positives? Token leaking? Should we strip Authorization on cross-host redirects?
5. Concurrency: any data races in Client/Provider?
6. Validation: gaps in Request.Validate (empty AspectRatio when Size also empty handled?). Provider 'auto' string handling. Token override behavior.
7. CLI flag aliases: do `-t` and `-token` interact correctly? What if both are passed? What about default-value collisions for aliased flags?
8. Stdin detection: does the os.Stdin.Stat ModeCharDevice trick work in all cases (e.g. when -p is given AND stdin is piped)?
9. Output path collisions: if user passes `-o foo.webp` with `-number-of-images 3`, do we overwrite? What if outputPath has no extension?
10. Test coverage gaps and any tests that might be brittle (timing, ports).
11. The replicateProvider.downloadBytes only sends Authorization for replicate.delivery / api.replicate.com — is that right for all CDN paths?
12. JSON decoding: any field shadowing, missing fields, error response shape mismatches?
13. Are we handling the OpenAI 'background: transparent' explicitly (gpt-image-2 doesn't support it)?
14. The OpenAI edits endpoint requires PNG with alpha for masks per docs — do we validate?
15. ResolveSize falls back to "auto" for unknown ratios — is that desirable, or should it error?

Return: a numbered list of issues, severity (CRITICAL/HIGH/MEDIUM/LOW), file:line, and a one-line proposed fix. End with a short "must-fix before ship" subset.