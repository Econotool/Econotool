"""Stata .do file parser for replication specification extraction.

Parses Stata .do files from replication packages and extracts:
- Variable construction commands (gen, replace, label)
- Sample restrictions (keep, drop, if conditions)
- Regression commands (reg, ivregress, xtreg, areg, reghdfe, etc.)
- Estimation options (robust, cluster, absorb, etc.)

The parser is intentionally tolerant of Stata syntax variations — it
extracts what it can and flags what it cannot parse for human review.

This is NOT a full Stata interpreter. It maps Stata estimation commands
to :class:`econtools.replication.spec.ColumnSpec` objects.

Public API
----------
parse_do_file   — parse a .do file → ParsedDoFile
do_to_columns   — extract ColumnSpec list from parsed commands
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Parsed representations
# ---------------------------------------------------------------------------

@dataclass
class StataCommand:
    """A single parsed Stata command."""

    raw: str                          # original text
    command: str                      # e.g. "reg", "ivregress", "gen"
    args: list[str] = field(default_factory=list)
    options: dict[str, str | bool] = field(default_factory=dict)
    if_condition: str = ""            # e.g. "if inlf == 1"
    in_range: str = ""                # e.g. "in 1/100"
    weight: str = ""                  # e.g. "[pw=wt]"
    line_number: int = 0
    parsed_ok: bool = True
    parse_notes: list[str] = field(default_factory=list)


@dataclass
class VariableConstruction:
    """A variable generation / transformation command."""

    target_var: str
    expression: str                   # Stata expression
    command: str                      # "gen", "egen", "replace"
    condition: str = ""               # if condition
    label: str = ""
    line_number: int = 0


@dataclass
class SampleCommand:
    """A sample restriction command (keep/drop/if)."""

    action: str                       # "keep", "drop"
    condition: str
    line_number: int = 0


@dataclass
class RegressionCommand:
    """A parsed regression / estimation command."""

    dep_var: str
    exog_vars: list[str]
    endog_vars: list[str] = field(default_factory=list)
    instruments: list[str] = field(default_factory=list)
    estimator: str = "ols"
    command: str = "reg"              # original Stata command
    cluster_var: str | None = None
    robust: bool = False
    absorb_var: str | None = None     # areg/reghdfe
    absorb_vars: list[str] = field(default_factory=list)
    weight_var: str | None = None
    weight_type: str = ""             # "pw", "aw", "fw", "iw"
    if_condition: str = ""
    panel_entity: str | None = None   # from xtset
    panel_time: str | None = None     # from xtset
    fe: bool = False
    re: bool = False
    options_raw: dict[str, str | bool] = field(default_factory=dict)
    line_number: int = 0
    parse_notes: list[str] = field(default_factory=list)


@dataclass
class ParsedDoFile:
    """Complete parsed representation of a .do file."""

    path: str
    raw_lines: list[str]
    commands: list[StataCommand] = field(default_factory=list)
    regressions: list[RegressionCommand] = field(default_factory=list)
    variable_constructions: list[VariableConstruction] = field(default_factory=list)
    sample_commands: list[SampleCommand] = field(default_factory=list)
    data_loads: list[str] = field(default_factory=list)       # use, import, insheet
    xtset: tuple[str, str] | None = None                      # (entity, time)
    globals_defined: dict[str, str] = field(default_factory=dict)
    locals_defined: dict[str, str] = field(default_factory=dict)
    unparsed_commands: list[str] = field(default_factory=list)
    parse_warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stata command aliases → econtools estimator names
# ---------------------------------------------------------------------------

_ESTIMATOR_MAP = {
    "reg": "ols",
    "regress": "ols",
    "areg": "ols",          # OLS with absorbed FE
    "reghdfe": "ols",       # high-dimensional FE
    "ivregress": "2sls",
    "ivreg2": "2sls",
    "ivreg": "2sls",
    "xtreg": "panel",       # determined by fe/re option
    "xtivreg": "2sls",
    "xtivreg2": "2sls",
    "xtivregress": "2sls",
    "probit": "probit",
    "logit": "logit",
    "tobit": "tobit",
    "poisson": "poisson",
    "nbreg": "nbreg",
    "newey": "ols",         # OLS with Newey-West SEs
    "newey2": "ols",
    "sureg": "sur",
    "reg3": "3sls",
    "gmm": "gmm",
    "xtabond": "ab_gmm",
    "xtabond2": "ab_gmm",
    "xtdpdsys": "sys_gmm",
    "didregress": "did",
    "diff": "did",
    "rdrobust": "rdd",
    "rd": "rdd",
}

# Commands that define variables
_GEN_COMMANDS = {"gen", "generate", "egen", "replace"}

# Commands that restrict samples
_SAMPLE_COMMANDS = {"keep", "drop"}

# Commands that load data
_LOAD_COMMANDS = {"use", "import", "insheet", "infile", "input"}

# Regression commands
_REG_COMMANDS = set(_ESTIMATOR_MAP.keys())


# ---------------------------------------------------------------------------
# Tokeniser: join continuation lines, strip comments
# ---------------------------------------------------------------------------

def _preprocess_do(text: str) -> list[tuple[int, str]]:
    """Preprocess .do file text into clean command lines with line numbers.

    Handles:
    - Line continuation (/// and /*...*/)
    - Comments (* at line start, // inline, /* block */)
    - Multiple commands on one line (separated by \\n but not ;)
    """
    lines = text.splitlines()
    result: list[tuple[int, str]] = []

    # First pass: remove block comments
    in_block = False
    cleaned: list[tuple[int, str]] = []
    for i, line in enumerate(lines, 1):
        if in_block:
            end = line.find("*/")
            if end >= 0:
                line = line[end + 2:]
                in_block = False
            else:
                continue

        # Remove block comments that start and end on same line
        while "/*" in line:
            start = line.find("/*")
            end = line.find("*/", start + 2)
            if end >= 0:
                line = line[:start] + " " + line[end + 2:]
            else:
                line = line[:start]
                in_block = True
                break

        cleaned.append((i, line))

    # Second pass: handle line continuations (///)
    merged: list[tuple[int, str]] = []
    accumulator = ""
    start_line = 0
    for lineno, line in cleaned:
        stripped = line.strip()

        # Skip pure comment lines
        if stripped.startswith("*"):
            continue

        # Remove inline // comments (but not ///)
        comment_pos = _find_inline_comment(stripped)
        if comment_pos >= 0:
            stripped = stripped[:comment_pos].rstrip()

        # Handle continuation
        if stripped.endswith("///"):
            if not accumulator:
                start_line = lineno
            accumulator += stripped[:-3] + " "
            continue

        if accumulator:
            accumulator += stripped
            merged.append((start_line, accumulator.strip()))
            accumulator = ""
        elif stripped:
            merged.append((lineno, stripped))

    if accumulator:
        merged.append((start_line, accumulator.strip()))

    return merged


def _find_inline_comment(line: str) -> int:
    """Find position of // comment (not /// continuation)."""
    i = 0
    in_quote = False
    while i < len(line) - 1:
        if line[i] == '"':
            in_quote = not in_quote
        elif not in_quote and line[i:i+2] == "//":
            # Check that this // is not part of a /// continuation
            if line[i:i+3] == "///":
                i += 3  # skip past the ///
                continue
            return i
        i += 1
    return -1


# ---------------------------------------------------------------------------
# Option parser
# ---------------------------------------------------------------------------

def _parse_options(option_str: str) -> dict[str, str | bool]:
    """Parse Stata comma-separated options into a dict.

    Examples:
        "robust cluster(state)" → {"robust": True, "cluster": "state"}
        "vce(cluster state)"    → {"vce": "cluster state"}
    """
    opts: dict[str, str | bool] = {}
    if not option_str.strip():
        return opts

    # Tokenise respecting parentheses
    tokens: list[str] = []
    current = ""
    depth = 0
    for ch in option_str:
        if ch == "(":
            depth += 1
            current += ch
        elif ch == ")":
            depth -= 1
            current += ch
        elif ch == " " and depth == 0:
            if current.strip():
                tokens.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        tokens.append(current.strip())

    for tok in tokens:
        m = re.match(r"(\w+)\((.+)\)", tok)
        if m:
            opts[m.group(1).lower()] = m.group(2).strip()
        else:
            opts[tok.lower()] = True

    return opts


# ---------------------------------------------------------------------------
# Command-level parsers
# ---------------------------------------------------------------------------

def _parse_if_condition(args_str: str) -> tuple[str, str]:
    """Split args_str into (args_before_if, if_condition).

    Returns (original, "") if no 'if' found.
    """
    # Match ' if ' not inside parentheses
    m = re.search(r"\bif\b\s+(.+?)(?:,|\[|$)", args_str, re.IGNORECASE)
    if m:
        before = args_str[:m.start()].strip()
        condition = m.group(1).strip()
        return before, condition
    return args_str, ""


def _parse_weight(args_str: str) -> tuple[str, str, str]:
    """Extract weight specification [pw=var] from args string.

    Returns (args_without_weight, weight_type, weight_var).
    """
    m = re.search(r"\[(pw|aw|fw|iw)\s*=\s*(\w+)\]", args_str, re.IGNORECASE)
    if m:
        cleaned = args_str[:m.start()] + args_str[m.end():]
        return cleaned.strip(), m.group(1).lower(), m.group(2)
    return args_str, "", ""


def _split_command_options(line: str) -> tuple[str, str]:
    """Split a Stata command line into (body, options) at the first top-level comma."""
    depth = 0
    for i, ch in enumerate(line):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            return line[:i].strip(), line[i+1:].strip()
    return line, ""


def _parse_varlist(s: str) -> list[str]:
    """Parse a space-separated variable list, handling i. and c. prefixes."""
    tokens = s.split()
    result = []
    for tok in tokens:
        # Remove factor variable notation: i.var, c.var, ib3.var, etc.
        cleaned = re.sub(r"^[ic](?:b\d+)?\.(.+)$", r"\1", tok)
        # Remove ## interaction notation
        if "#" in cleaned:
            parts = re.split(r"##?", cleaned)
            for p in parts:
                p = re.sub(r"^[ic](?:b\d+)?\.(.+)$", r"\1", p)
                if p:
                    result.append(p)
        else:
            result.append(cleaned)
    return result


def _parse_regression(line: str, lineno: int,
                      xtset: tuple[str, str] | None = None) -> RegressionCommand:
    """Parse a Stata regression command into a RegressionCommand."""
    body, option_str = _split_command_options(line)
    options = _parse_options(option_str)

    # Extract command name
    tokens = body.split()
    cmd = tokens[0].lower()
    rest = " ".join(tokens[1:])

    # Extract weight
    rest, wt_type, wt_var = _parse_weight(rest)

    # Extract if condition
    rest, if_cond = _parse_if_condition(rest)

    notes: list[str] = []
    dep_var = ""
    exog: list[str] = []
    endog: list[str] = []
    instruments: list[str] = []
    absorb: list[str] = []
    cluster_var: str | None = None
    is_fe = False
    is_re = False

    # Determine estimator
    estimator = _ESTIMATOR_MAP.get(cmd, "unknown")

    # --- IV commands: dep_var (endog = instruments) exog ---
    if cmd in ("ivregress", "ivreg2", "ivreg", "xtivreg", "xtivreg2", "xtivregress"):
        # ivregress 2sls dep_var exog (endog = instruments)
        # ivreg2 dep_var exog (endog = instruments)
        sub_cmd_rest = rest

        # For ivregress, first token is method (2sls, liml, gmm)
        if cmd == "ivregress":
            method_tokens = sub_cmd_rest.split(None, 1)
            if len(method_tokens) >= 2:
                method = method_tokens[0].lower()
                sub_cmd_rest = method_tokens[1]
                if method == "liml":
                    estimator = "liml"
                elif method == "gmm":
                    estimator = "gmm"

        # Parse parenthesised instrument block
        m = re.search(r"\((.+?)\)", sub_cmd_rest)
        if m:
            iv_block = m.group(1)
            before_paren = sub_cmd_rest[:m.start()].strip()
            after_paren = sub_cmd_rest[m.end():].strip()

            # dep_var is first token before paren
            before_tokens = before_paren.split()
            if before_tokens:
                dep_var = before_tokens[0]
                exog = _parse_varlist(" ".join(before_tokens[1:]))
            # After paren may have more exog vars
            if after_paren:
                exog.extend(_parse_varlist(after_paren))

            # Parse instrument block: "endog = instruments" or just "instruments"
            if "=" in iv_block:
                endog_str, inst_str = iv_block.split("=", 1)
                endog = _parse_varlist(endog_str.strip())
                instruments = _parse_varlist(inst_str.strip())
            else:
                instruments = _parse_varlist(iv_block)
        else:
            notes.append("IV command without parenthesised instruments")
            all_vars = _parse_varlist(sub_cmd_rest)
            if all_vars:
                dep_var = all_vars[0]
                exog = all_vars[1:]

    # --- Panel commands ---
    elif cmd in ("xtreg",):
        all_vars = _parse_varlist(rest)
        if all_vars:
            dep_var = all_vars[0]
            exog = all_vars[1:]
        if options.get("fe"):
            is_fe = True
            estimator = "fe"
        elif options.get("re"):
            is_re = True
            estimator = "re"
        elif options.get("be"):
            estimator = "between"
        elif options.get("mle"):
            estimator = "re"
            notes.append("MLE random effects")
        else:
            estimator = "fe"  # default for xtreg
            notes.append("No fe/re option specified, assuming FE")

    # --- areg / reghdfe (absorbed FE) ---
    elif cmd in ("areg", "reghdfe"):
        all_vars = _parse_varlist(rest)
        if all_vars:
            dep_var = all_vars[0]
            exog = all_vars[1:]

        if "absorb" in options:
            absorb_str = options["absorb"]
            if isinstance(absorb_str, str):
                absorb = _parse_varlist(absorb_str)
        estimator = "fe"

    # --- Standard regression ---
    elif cmd in ("reg", "regress", "probit", "logit", "tobit", "poisson",
                 "nbreg", "newey", "newey2"):
        all_vars = _parse_varlist(rest)
        if all_vars:
            dep_var = all_vars[0]
            exog = all_vars[1:]

        if cmd in ("newey", "newey2"):
            estimator = "ols"
            notes.append("Newey-West SEs")

    else:
        # Fallback: try to parse as dep_var varlist
        all_vars = _parse_varlist(rest)
        if all_vars:
            dep_var = all_vars[0]
            exog = all_vars[1:]
        notes.append(f"Unrecognised command '{cmd}' — parsed as generic regression")

    # --- Extract SE/clustering from options ---
    robust = bool(options.get("robust") or options.get("r"))

    # vce() option
    vce_str = options.get("vce", "")
    if isinstance(vce_str, str):
        vce_lower = vce_str.lower()
        if "cluster" in vce_lower:
            m = re.match(r"cluster\s+(\w+)", vce_lower)
            if m:
                cluster_var = m.group(1)
        elif vce_lower in ("robust", "hc1"):
            robust = True
        elif "hac" in vce_lower or "newey" in vce_lower:
            notes.append(f"HAC/Newey-West via vce({vce_str})")

    # cluster() option (older syntax)
    cl_str = options.get("cluster", "")
    if isinstance(cl_str, str) and cl_str:
        cluster_var = cl_str.split()[0]

    # Panel info from xtset
    entity_col = None
    time_col = None
    if xtset:
        entity_col, time_col = xtset
    if cmd.startswith("xt"):
        if not entity_col:
            notes.append("xt-command without prior xtset — entity/time unknown")

    return RegressionCommand(
        dep_var=dep_var,
        exog_vars=exog,
        endog_vars=endog,
        instruments=instruments,
        estimator=estimator,
        command=cmd,
        cluster_var=cluster_var,
        robust=robust,
        absorb_var=absorb[0] if len(absorb) == 1 else None,
        absorb_vars=absorb,
        weight_var=wt_var if wt_var else None,
        weight_type=wt_type,
        if_condition=if_cond,
        panel_entity=entity_col,
        panel_time=time_col,
        fe=is_fe,
        re=is_re,
        options_raw=options,
        line_number=lineno,
        parse_notes=notes,
    )


def _parse_gen_command(line: str, lineno: int) -> VariableConstruction | None:
    """Parse gen/egen/replace commands."""
    tokens = line.split()
    cmd = tokens[0].lower()

    if cmd in ("gen", "generate"):
        # gen [type] varname = expression [if ...]
        rest = " ".join(tokens[1:])
        # Skip optional type declaration
        rest = re.sub(r"^(byte|int|long|float|double|str\d+)\s+", "", rest)
        m = re.match(r"(\w+)\s*=\s*(.+?)(?:\s+if\s+(.+))?$", rest, re.IGNORECASE)
        if m:
            return VariableConstruction(
                target_var=m.group(1),
                expression=m.group(2).strip(),
                command=cmd,
                condition=m.group(3) or "",
                line_number=lineno,
            )

    elif cmd == "egen":
        rest = " ".join(tokens[1:])
        rest = re.sub(r"^(byte|int|long|float|double|str\d+)\s+", "", rest)
        m = re.match(r"(\w+)\s*=\s*(.+?)(?:\s+if\s+(.+))?$", rest, re.IGNORECASE)
        if m:
            return VariableConstruction(
                target_var=m.group(1),
                expression=m.group(2).strip(),
                command=cmd,
                condition=m.group(3) or "",
                line_number=lineno,
            )

    elif cmd == "replace":
        rest = " ".join(tokens[1:])
        m = re.match(r"(\w+)\s*=\s*(.+?)(?:\s+if\s+(.+))?$", rest, re.IGNORECASE)
        if m:
            return VariableConstruction(
                target_var=m.group(1),
                expression=m.group(2).strip(),
                command=cmd,
                condition=m.group(3) or "",
                line_number=lineno,
            )

    elif cmd == "label" and len(tokens) >= 4 and tokens[1].lower() == "variable":
        # label variable varname "label text"
        var = tokens[2]
        label_text = " ".join(tokens[3:]).strip('"').strip("'")
        return VariableConstruction(
            target_var=var,
            expression="",
            command="label",
            label=label_text,
            line_number=lineno,
        )

    return None


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_do_file(path: str | Path) -> ParsedDoFile:
    """Parse a Stata .do file into structured components.

    Parameters
    ----------
    path:
        Path to the .do file.

    Returns
    -------
    ParsedDoFile
        Structured representation of all commands found.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    raw_lines = text.splitlines()
    commands = _preprocess_do(text)

    result = ParsedDoFile(
        path=str(path),
        raw_lines=raw_lines,
    )

    xtset: tuple[str, str] | None = None

    for lineno, line in commands:
        tokens = line.split()
        if not tokens:
            continue

        cmd = tokens[0].lower()

        # Remove leading "quietly" / "capture" / "noisily"
        while cmd in ("quietly", "qui", "capture", "cap", "noisily", "noi"):
            tokens = tokens[1:]
            if not tokens:
                break
            cmd = tokens[0].lower()
            line = " ".join(tokens)
        if not tokens:
            continue

        # xtset
        if cmd == "xtset":
            parts = tokens[1:]
            if len(parts) >= 2:
                xtset = (parts[0], parts[1])
                result.xtset = xtset
            elif len(parts) == 1:
                xtset = (parts[0], "")
                result.xtset = xtset
            continue

        # tsset (alias for xtset in time-series context)
        if cmd == "tsset":
            parts = tokens[1:]
            if len(parts) >= 2:
                xtset = (parts[0], parts[1])
                result.xtset = xtset
            continue

        # Data loads
        if cmd in _LOAD_COMMANDS:
            result.data_loads.append(line)
            continue

        # Variable generation
        if cmd in _GEN_COMMANDS or (cmd == "label" and len(tokens) >= 2
                                     and tokens[1].lower() == "variable"):
            vc = _parse_gen_command(line, lineno)
            if vc:
                result.variable_constructions.append(vc)
            continue

        # Sample restrictions
        if cmd in _SAMPLE_COMMANDS:
            rest = " ".join(tokens[1:])
            result.sample_commands.append(SampleCommand(
                action=cmd,
                condition=rest,
                line_number=lineno,
            ))
            continue

        # Globals and locals
        if cmd == "global":
            rest = " ".join(tokens[1:])
            m = re.match(r"(\w+)\s+(.*)", rest)
            if m:
                result.globals_defined[m.group(1)] = m.group(2)
            continue
        if cmd == "local":
            rest = " ".join(tokens[1:])
            m = re.match(r"(\w+)\s+(.*)", rest)
            if m:
                result.locals_defined[m.group(1)] = m.group(2)
            continue

        # Regression commands
        if cmd in _REG_COMMANDS:
            reg = _parse_regression(line, lineno, xtset=xtset)
            result.regressions.append(reg)
            result.commands.append(StataCommand(
                raw=line, command=cmd, line_number=lineno,
            ))
            continue

        # Post-estimation (eststo, estimates store, outreg2, etc.)
        if cmd in ("eststo", "estimates", "outreg2", "esttab", "estout",
                    "estadd", "test", "testparm", "lincom"):
            result.commands.append(StataCommand(
                raw=line, command=cmd, line_number=lineno,
            ))
            continue

        # Everything else
        result.unparsed_commands.append(line)

    return result


def do_to_column_specs(parsed: ParsedDoFile) -> list:
    """Convert parsed regression commands to ColumnSpec objects.

    Returns a list of :class:`econtools.replication.spec.ColumnSpec` objects,
    one per regression command found.
    """
    from econtools.replication.spec import ColumnSpec

    columns = []
    for i, reg in enumerate(parsed.regressions, 1):
        # Determine cov_type
        if reg.cluster_var:
            cov_type = "cluster"
        elif reg.robust:
            cov_type = "HC1"
        else:
            cov_type = "classical"

        col = ColumnSpec(
            column_id=f"({i})",
            dep_var=reg.dep_var,
            exog_vars=reg.exog_vars,
            endog_vars=reg.endog_vars,
            instruments=reg.instruments,
            estimator=reg.estimator,
            cov_type=cov_type,
            cluster_var=reg.cluster_var,
            absorb_vars=reg.absorb_vars,
            entity_col=reg.panel_entity,
            time_col=reg.panel_time,
            weights_var=reg.weight_var,
            sample_restriction=reg.if_condition or None,
            label=f"Line {reg.line_number}: {reg.command}",
            notes="; ".join(reg.parse_notes) if reg.parse_notes else "",
        )
        columns.append(col)

    return columns
