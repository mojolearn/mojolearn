"""Exact generated-root/member-export proof shared by retained development adapters."""
import ast
import hashlib

from release_ledger_admission import require


def entry_proof(tree, fields):
    """Bind the generated root to the identical canonical member exports.

    A development root adds its private membership module. It exports each
    public PyInit using the same import/call body as the per-binding entry.
    Only that canonical generated wrapper may be absent from the group closure.
    No substantive module source or changed entry body is exempted.
    """
    source = tree.text("tools/merged_bindings.py")
    constants = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {"ENTRY", "ROOT_HEAD", "ROOT_MEMBER", "ROOT_TAIL"}:
                constants[name] = ast.literal_eval(node.value)
    require(len(constants) == 4, "tested wrapper generator templates are unsupported")
    stem, members = fields["stem"], fields["members"]
    root = constants["ROOT_HEAD"].format(stem=stem)
    root += "".join(f"from {name} import build_{name}\n" for name in members)
    entries = {}
    for name in members:
        entry = constants["ENTRY"].format(name=name)
        member = constants["ROOT_MEMBER"].format(name=name)
        require(member.strip() in entry and f"from {name} import build_{name}" in entry,
                "public entry and generated member body differ")
        path = f"bindings/entry/entry_{name}.mojo"
        require(tree.text(path) == entry, "public entry is not the tested canonical wrapper: " + path)
        entries[path] = hashlib.sha256(entry.encode()).hexdigest()
        root += member
    root += constants["ROOT_TAIL"].format(stem=stem, names=",".join(members))
    require(hashlib.sha256(root.encode()).hexdigest() == fields.get("root_sha256"),
            "generated development root differs from recorded bytes")
    return entries

