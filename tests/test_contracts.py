"""Tests for the formal, versioned agent contracts."""

from __future__ import annotations

import pytest

from mars.agents.contracts import (
    CONTRACT_VERSION,
    CONTRACTS,
    SCHEMA_MODELS,
    ContractError,
    contract_for,
    contract_manifest,
    schema_model,
    validate_contracts,
)
from mars.agents.profiles import PROFILES


def test_every_agent_profile_has_a_contract() -> None:
    assert set(PROFILES) == set(CONTRACTS)
    assert validate_contracts() == []


def test_contracts_are_versioned_and_complete() -> None:
    for key, contract in CONTRACTS.items():
        assert contract.key == key
        assert contract.version == CONTRACT_VERSION
        assert contract.responsibilities
        assert contract.inputs
        assert contract.outputs
        assert contract.failure_modes
        assert contract.invariants
        assert contract.output_schema


def test_contract_for_unknown_agent_raises() -> None:
    with pytest.raises(ContractError, match="no behaviour contract"):
        contract_for("ghost")


def test_structured_output_schemas_resolve_to_models() -> None:
    assert schema_model("ResearchPlan") is not None
    assert schema_model("FindingBundle") is not None
    assert schema_model("AuditBundle") is not None
    assert schema_model("EvaluationResult") is not None
    assert schema_model("Dossier") is None
    for contract in CONTRACTS.values():
        if contract.output_schema in SCHEMA_MODELS:
            assert schema_model(contract.output_schema) is not None


def test_contract_manifest_serialises_every_agent() -> None:
    manifest = contract_manifest()
    assert manifest["version"] == CONTRACT_VERSION
    assert set(manifest["agents"]) == set(CONTRACTS)
    assert manifest["agents"]["writer"]["output_schema"] == "SectionMarkdown"
    assert "failure_modes" in manifest["agents"]["auditor"]


def test_contracts_name_the_domain_invariants() -> None:
    assert any("Agentic AI" in item for item in CONTRACTS["director"].invariants)
    assert any("Agentic AI" in item for item in CONTRACTS["researcher"].invariants)
    assert any("[S#]" in item for item in CONTRACTS["writer"].invariants)
