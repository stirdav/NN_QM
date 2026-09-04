# Onboarding: working on QuantumDynamics with an AI assistant

A short, practical guide for new contributors. Nothing physics-specific here —
just how to work with the AI assistant, how we work on this repo, and how we
run Julia.

## Working with the AI assistant (Claude Code or similar)

- **`CLAUDE.md` is the assistant's briefing.** It is loaded automatically at
  the start of every session. Read it yourself too — it is the fastest
  overview of the codebase. If you catch yourself re-explaining the same
  context every session, that context belongs in `CLAUDE.md` or `design.md`.
- **`design.md` is the "why".** Ask the assistant to read it before any
  non-trivial change under `src/`. It records decisions and load-bearing
  conventions that are not obvious from the code alone.
- **Give it a task, not just a nudge.** State the use case, the constraints,
  and ask it to propose an approach before it writes code. Describe the
  problem you have, not the solution you imagine.
- **Review everything it writes.** Read the diff the way you would review a
  colleague's pull request. The assistant is fluent and confident even when
  it is wrong. You own the commit.
- **Make it run things before you trust them.** Ask it to run the test suite
  or the relevant example (through Kaimon — see below). Do not accept "this
  should work" as evidence.
- **Use it as a critic.** It is good at finding edge cases and design
  problems when you explicitly ask it to be adversarial — point it at a
  commit or at the test suite and ask it to poke holes.
- **One task per session.** Long conversations drift, lose track of earlier
  decisions, and cost more. Finish a task, start a fresh session for the next.
- **It has a persistent memory** for your preferences and for corrections you
  give it. If it does something you dislike, say "remember: ..." and it will
  carry that into future sessions.
- **Do not let it invent numbers or results.** Anything quantitative must come
  from an actual run, not from the model. Check.

### Example prompts

These are the kinds of prompts that have worked well on this project. Adapt
the specifics; keep the shape.

**Proposing a new feature** — lead with the use case and ask to discuss
before any code is written:

> I would like to add a new functionality for general convenience. It is a
> common use case that I will be executing some long-running simulation, and
> that I would like to do post-processing or analysis later. Therefore, one
> should be able to save the result of a simulation as Julia-native data
> objects. This involves saving the states/trajectories over time (possibly
> coarse-grained in time) together with the most important parameters for
> that run. Let's discuss what needs to go into this.

**Reviewing a commit** — ask for criticism explicitly, and tell it the docs
are not gospel:

> I want you to critically evaluate the changes in the latest commit in this
> repository. Think about the choice of design and structure. Be critical and
> think about potential problems or edge cases down the road. Keep in mind
> that there were made changes in design.md and CLAUDE.md in the same commit,
> so they are not an absolute single source of truth. Feel free to interview
> me about intended behavior of the project.

**Auditing the test suite** — push it to check coverage against the actual
source, not just that tests pass:

> I want you to critically go through the current test suite. Make sure that
> the tests actually cover the code in the repo, and are not too artificial
> or constructed. Go through the source code and look out for edge cases that
> are not caught by the tests. Feel free to interview me about the intended
> behaviour.

## How we work on this repo structurally

- **Thin layer, not a re-implementation.** QuantumDynamics sits on top of
  `QuantumOptics.jl` and only adds organization (named subsystems, composite
  systems, Hamiltonian/dissipator recipes, one `evolve` entry point). If you
  are about to reimplement solver or Hilbert-space machinery, stop.
- **Pull before you start.** Run `git fetch` (or `git pull`) at the start of
  every work session — other people push to `main` regularly, and starting
  from a stale local copy leads to avoidable conflicts and duplicated work.
- **Committing directly to `main` is fine by default.** The team is small and
  this is a research framework, so a short-lived breakage on `main` is
  acceptable. Fetch and rebase onto `origin/main` before pushing to keep
  history linear. Use a short-lived branch and a pull request when you want a
  second person to review before the change lands — for example larger
  changes, anything touching core numerics, or a new contributor's first few
  changes.
- **If a push breaks `main`:** revert the offending commit
  (`git revert <sha>`) and push the revert so `main` is green again, let the
  others know, then redo the change properly. Do not leave a known-broken
  `main` for someone else to discover.
- **Run the full test suite before every push.** 137 tests today; keep them
  green. New behaviour in `src/` ships with a test in `test/runtests.jl`.
- **A `src/` change usually touches more than one file:** the code, its test,
  and often `design.md` and `CLAUDE.md`. The docs are part of the change, not
  a follow-up.
- **Comments explain the code, not the conversation.** Do not leave notes like
  "as discussed" or "AI-validated" — keep the finding, drop the provenance
  (see commit `3d77bd8`).
- **Commit messages are prose.** A one-line summary, then paragraphs on *why*
  and what was traded off — look at recent `git log`.
- **Examples are self-contained.** Each `examples/*/` directory has its own
  `Project.toml` and `README.md` and points back at the package through the
  `[sources]` table. They act as integration tests and as the place to
  prototype a feature before it is promoted into `src/`.

## Working with Julia (via Kaimon)

We run all Julia in this repo through [Kaimon](https://github.com/kahliburke/Kaimon.jl),
an MCP server that gives the AI assistant a live Julia REPL that you share and
can watch from a terminal dashboard.

- **Install once** (needs Julia 1.12+). In a Julia REPL:
  ```julia
  ]app add Kaimon
  ```
  This puts a `kaimon` launcher in `~/.julia/bin/`, which is not on your
  `PATH` by default. Add it by putting this line in your shell startup file
  (`~/.zshrc` on macOS, `~/.bashrc` on most Linux):
  ```sh
  export PATH="$HOME/.julia/bin:$PATH"
  ```
  then open a new terminal (or `source ~/.zshrc`) and check `which kaimon`
  resolves.
- **Run it** by launching the dashboard and leaving it open while you work:
  ```sh
  kaimon
  ```
  The first run walks you through a short setup (security mode, API key,
  port — the default is `2828`). After that it opens the TUI dashboard that
  shows every session, eval, and test run the assistant triggers.
- **Connect your editor's assistant to it once:** from the dashboard's Config
  tab press `i` to write the MCP config for Claude Code / Cursor / VS Code.
  Verify with `/mcp` inside Claude Code — `kaimon` should show as connected.
- The assistant loads its `kaimon` skill automatically for any Julia work and
  drives the server itself; you mostly just keep the dashboard running.
- **Watch for silent fallbacks.** If the assistant can't reach Kaimon — the
  dashboard isn't running, the session won't start, a project isn't on the
  allowed list — it will often just run `julia` directly in the terminal
  instead, without flagging that it switched. You can let that slide for a
  quick one-off, but in general it's better to interrupt and ask why it
  isn't going through Kaimon: the reason is usually a small setup problem
  worth fixing once, and a bare terminal run is one you can't see in the
  dashboard and doesn't get Revise hot-reload.
