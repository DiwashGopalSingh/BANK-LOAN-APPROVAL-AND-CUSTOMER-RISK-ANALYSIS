# AI Workflow Rules

Rules for any AI agent (Claude, or the Antigravity in-IDE agent) working on this project.

1. **Stay inside the scope in `project-overview.md`.** Don't add authentication, deployment infra, or extra features not listed there without checking first.
2. **Work in small, verifiable steps.** Build and confirm one piece at a time (e.g., preprocessing → confirm output shape, then move to training) rather than generating the whole pipeline in one pass.
3. **Never fabricate results.** Report actual metrics from actual runs — don't estimate or invent accuracy/precision numbers.
4. **Explain non-obvious choices.** If a design decision differs from `architecture-context.md`, say why before making it.
5. **Keep code beginner-readable**, per `code-standards.md` — this project is also a learning exercise, not just a deliverable.
6. **Don't skip EDA.** Understand the data (distributions, missing values, class balance) before jumping to modeling.
7. **Ask before restructuring.** If a change would affect the folder structure or the approval/risk separation in `architecture-context.md`, flag it first rather than silently refactoring.
8. **Update `progress-tracker.md`** as milestones are completed.
