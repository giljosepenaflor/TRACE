# TRACE project context

This repository supports the user's undergraduate thesis, **TRACE: A Lightweight Dual-View Framework for Backdoor Activation Detection in Selected Open Source Large Language Models**.

Before planning or changing this project, read `docs/thesis/PROJECT_CONTEXT.md`. The preserved user-provided thesis is `docs/thesis/source-draft.txt`; consult the relevant sections when details matter.

Read `docs/DEVELOPMENT_STATUS.md` and `README.md` for the current prototype, verified commands, and remaining development milestones.

Use the thesis as the basis for implementation, experiments, documentation, and explanations. Follow later explicit user corrections when the methodology changes, and keep the project context current. Distinguish proposed methodology, implemented functionality, and measured results. Never invent experimental findings or treat a triggered input alone as a confirmed backdoor activation.

Preserve the study's scope: a lightweight dual-view detector, separate detectors per target model, controlled experiments, strict partition isolation, and a minimal monitoring interface. Detection does not remove or modify backdoors. Do not claim cross-model generalization or universal backdoor detection without experiments supporting it.
