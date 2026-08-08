import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs" / "DATA_COLLECTION_PROTOCOL.md"
SCHEMA = ROOT / "docs" / "schemas" / "real_data_manifest.schema.json"
MANIFEST_TEMPLATE = (
    ROOT / "docs" / "templates" / "real_data_manifest.template.json"
)
EVENT_LABELS_TEMPLATE = (
    ROOT / "docs" / "templates" / "event_labels.template.csv"
)

EXPECTED_EVENT_COLUMNS = [
    "experiment_id",
    "event_id",
    "sensor_id",
    "event_type",
    "start_time_earliest",
    "start_time_latest",
    "end_time_earliest",
    "end_time_latest",
    "label_status",
    "confidence",
    "expert_comment",
    "evidence_reference",
]


def test_protocol_states_current_evidence_gap_and_no_fake_data():
    text = PROTOCOL.read_text(encoding="utf-8")
    normalized_text = " ".join(text.replace("**", "").split())

    required_phrases = [
        "не являются реальным набором данных",
        "несколько независимо проведённых реальных экспериментов",
        "экспертная разметка",
        "publication_permission",
        "по experiment_id",
        "untouched test",
        "SHA-256",
        "нельзя добавлять в GitHub",
    ]
    for phrase in required_phrases:
        assert phrase in normalized_text


def test_manifest_schema_requires_provenance_split_and_safety_fields():
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))

    assert schema["$schema"].endswith("2020-12/schema")
    assert schema["additionalProperties"] is False
    assert {
        "dataset_id",
        "dataset_version",
        "owner_role",
        "license",
        "publication_permission",
        "split_policy",
        "experiments",
    }.issubset(schema["required"])

    split_policy = schema["$defs"]["split_policy"]
    assert split_policy["properties"]["method"]["const"] == "by_experiment"
    assert (
        split_policy["properties"]["final_test_untouched"]["const"]
        is True
    )

    experiment_required = set(schema["$defs"]["experiment"]["required"])
    assert {
        "experiment_id",
        "split",
        "measurement_file",
        "measurement_sha256",
        "events_file",
        "sensor_ids",
        "source_temperature_unit",
        "sampling",
        "time",
        "normal_operation",
        "expert_labeling",
        "data_quality",
        "provenance",
        "access_classification",
    }.issubset(experiment_required)


def test_manifest_template_is_explicitly_non_real_and_split_by_experiment():
    manifest = json.loads(MANIFEST_TEMPLATE.read_text(encoding="utf-8"))

    assert manifest["template_only"] is True
    assert manifest["publication_permission"] == "no_publication"
    assert manifest["split_policy"] == {
        "method": "by_experiment",
        "final_test_untouched": True,
        "frozen_at": "1970-01-01T00:00:00Z",
        "rationale": (
            "REPLACE_WITH_SPLIT_RATIONALE_BEFORE_MODEL_EVALUATION"
        ),
    }

    experiments = manifest["experiments"]
    assert {item["split"] for item in experiments} == {
        "train",
        "validation",
        "test",
    }
    experiment_ids = [item["experiment_id"] for item in experiments]
    assert len(experiment_ids) == len(set(experiment_ids))
    assert all(item.startswith("REPLACE_") for item in experiment_ids)
    assert all(
        item["measurement_file"].startswith("REPLACE_WITH_PRIVATE_PATH/")
        for item in experiments
    )


def test_event_label_template_has_uncertain_boundary_fields_and_no_rows():
    with EVENT_LABELS_TEMPLATE.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    assert rows == [EXPECTED_EVENT_COLUMNS]


def test_data_document_links_collection_protocol():
    data_doc = (ROOT / "docs" / "DATA.md").read_text(encoding="utf-8")

    assert (
        "[протоколу сбора независимых реальных данных]"
        "(DATA_COLLECTION_PROTOCOL.md)"
        in data_doc
    )
    assert "не считается\nнезависимо размеченным test-набором" in data_doc