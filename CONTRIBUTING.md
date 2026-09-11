<!--
SPDX-FileCopyrightText: 2026 Contributors to the sundael project.

SPDX-License-Identifier: MPL-2.0
-->

# Contributing

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

## Ways of contributing

Contribution does not necessarily mean committing code to the repository. 
We recognize different levels of contributions as shown below in increasing order of dedication:

1. Test and use the project. Give feedback on the user experience or suggest new features.
2. Report bugs or security vulnerability
3. Fix bugs.
4. Improve the project with developing new features.

## Community Guidelines

This project follows the following [Code of Conduct](CODE_OF_CONDUCT.md).

## Git Commit Guidelines

This project uses [Semantic Versioning](https://semver.org). We use commit
messages to automatically determine the version bumps using [GitVersion](https://gitversion.net/docs/), so they should adhere to
the conventions of [Conventional Commits (v1.0.0)](https://www.conventionalcommits.org/en/v1.0.0/).

### Automatic versioning

We use [Semantic Release](https://semantic-release.org/) for versioning based on
conventional commits. If needed, you can use `semver: none` in the commit message to skip a version bump. Otherwise:

- Commit messages starting with `fix:` trigger a patch version bump
- Commit messages starting with `feat:` trigger a minor version bump
- Commit messages with a footer `BREAKING CHANGE:` trigger a major version bump.

## Code style

### Introduction
This section outlines the coding style and conventions to be followed when writing Python code in this project. Consistency in code style enhances readability and maintainability. 
These guidelines are based on the [Software Delivery Python guidelines](https://alliander.atlassian.net/wiki/spaces/SOF/pages/3769106465/Python+guidelines).
Adherence to these coding style guidelines will lead to a more consistent and maintainable codebase. In cases where these guidelines conflict with existing code, follow the existing style to maintain consistency within the project.

### PEP8
This project uses the PEP 8 Style Guide for Python Code. For all details about the various conventions please refer to: [PEP 8](https://peps.python.org/pep-0008/)

### General Guidelines
- Aim for code simplicity, readability, and maintainability.
- Use the SonarLint Plugin connected to SonarCloud for development
- Linters: 
  - ruff for (Black-compatible) formatting, sorting of imports and additional code fixes and linting
  - mypy for type checking
- Before committing code, run automated tests and code quality checks.
- Logic in the code should have >80% test coverage.

### Indentation
- Use 4 spaces for indentation. Do not use tabs.

### Naming Conventions
- Use `snake_case` for variable and function names.
- Use `CamelCase` for class names.
- Prefix private variables and functions with an underscore (e.g., `_private_var`).
- Use descriptive and meaningful variable and function names.

### Comments
- Write clear and concise comments for non-trivial code sections.
- Use [Google Style Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) to document all classes and functions.
- Avoid unnecessary or redundant comments.
  
### Function and Method Definitions
- Use a blank line to separate logical sections within a function.
- Keep function and method lengths reasonable; aim for clarity and maintainability.

### String Formatting
- Prefer f-strings (formatted string literals) for string formatting.
- Use double quotes for string literals, but be consistent throughout the codebase.

### Exception Handling
- Use specific exception types whenever possible rather than catching generic `Exception`.
- Handle exceptions gracefully with meaningful error messages.
- Avoid using bare `except:` clauses.

