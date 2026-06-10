from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import BRICS, Recap


MIN_FRAGMENT_ATOMS = 2
MAX_FALLBACK_FRAGMENTS = 10
FALLBACK_FRAGMENT_PATTERNS = [
    (Chem.MolFromSmarts("[C;X3](=O)-[N;X3]"), (0, 2)),
    (Chem.MolFromSmarts("[C;X3](=O)-[O;X2]"), (0, 2)),
    (Chem.MolFromSmarts("[S;X2]-[S;X2]"), (0, 1)),
]


def _canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None


def _fallback_fragments(mol: Chem.Mol) -> set[str]:
    bond_ids = set()
    for pattern, cut_pair in FALLBACK_FRAGMENT_PATTERNS:
        for match in mol.GetSubstructMatches(pattern):
            bond = mol.GetBondBetweenAtoms(match[cut_pair[0]], match[cut_pair[1]])
            if bond is not None:
                bond_ids.add(bond.GetIdx())
    if not bond_ids:
        return {Chem.MolToSmiles(mol, canonical=True)}

    fragmented = Chem.FragmentOnBonds(mol, sorted(bond_ids), addDummies=False)
    fragments = set()
    for fragment in Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=False):
        if fragment.GetNumAtoms() < MIN_FRAGMENT_ATOMS:
            continue
        try:
            Chem.SanitizeMol(fragment)
        except Exception:
            pass
        fragment_smiles = Chem.MolToSmiles(fragment, canonical=True)
        if fragment_smiles:
            fragments.add(fragment_smiles)
    if not fragments:
        fragments.add(Chem.MolToSmiles(mol, canonical=True))
    return set(sorted(fragments, key=len, reverse=True)[:MAX_FALLBACK_FRAGMENTS])


def extract_substructures(
    smiles: str,
    method: str = "brics",
    max_smiles_length: int = 300,
    max_atoms: int = 200,
) -> list[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    fragments: set[str] = set()
    use_fallback = (
        method.lower() == "brics"
        and (len(smiles) > max_smiles_length or mol.GetNumAtoms() > max_atoms)
    )
    if use_fallback:
        fragments = _fallback_fragments(mol)
    else:
        try:
            if method.lower() == "brics":
                fragments = set(BRICS.BRICSDecompose(mol, returnMols=False))
            elif method.lower() == "recap":
                tree = Recap.RecapDecompose(mol)
                fragments = set(tree.GetLeaves().keys()) if tree is not None else set()
            else:
                raise ValueError("decomposition must be 'brics' or 'recap'")
        except Exception:
            fragments = _fallback_fragments(mol)

    cleaned = {_canonical(fragment) for fragment in fragments}
    cleaned.discard(None)
    if not cleaned:
        cleaned = {Chem.MolToSmiles(mol, canonical=True)}
    return sorted(cleaned)


def match_fragment_atoms(mol: Chem.Mol, fragment_smiles: str) -> list[int]:
    query = Chem.MolFromSmiles(fragment_smiles)
    if query is None:
        return []
    dummy = Chem.MolFromSmarts("[#0]")
    query = Chem.DeleteSubstructs(query, dummy)
    if query.GetNumAtoms() == 0:
        return []
    matches = mol.GetSubstructMatches(query)
    return sorted({idx for match in matches for idx in match})
