import math

import pytest

from proseeval import features, rules
from proseeval.normalize import LeakageReport, leakage_report, normalize


class TestNormalize:
    def test_removes_detached_clitics(self):
        assert normalize("You ca n't do that .") == "You can't do that."

    def test_removes_space_before_punctuation(self):
        assert normalize("Hello , world ; yes : no !") == "Hello, world; yes: no!"

    def test_rejoins_tokenizer_split_hyphens(self):
        assert normalize("Kim Jong - Un") == "Kim Jong-Un"

    def test_preserves_em_dashes(self):
        # Em dashes are authored style, not a tokenizer artifact.
        assert "—" in normalize("One thing—another thing.")

    def test_preserves_real_contractions(self):
        assert normalize("You can't do that.") == "You can't do that."

    def test_preserves_sentence_boundaries(self):
        text = "First. Second. Third."
        assert normalize(text).count(".") == 3

    def test_collapses_runs_of_blank_lines(self):
        assert normalize("a\n\n\n\n\nb") == "a\n\nb"

    def test_empty_input(self):
        assert normalize("") == ""

    def test_idempotent(self):
        text = "It ca n't be , he said ."
        assert normalize(normalize(text)) == normalize(text)


class TestLeakage:
    def test_detects_one_sided_artifact(self):
        human = ["He ca n't go ."] * 10
        machine = ["He can't go."] * 10
        reports = {r.artifact: r for r in leakage_report(machine, human)}
        assert reports["space_before_punct"].separation == pytest.approx(1.0)
        assert reports["detached_clitic"].separation == pytest.approx(1.0)

    def test_no_artifact_means_no_separation(self):
        clean = ["A clean sentence."] * 5
        for r in leakage_report(clean, clean):
            assert r.separation == 0.0

    def test_handles_empty_corpus(self):
        for r in leakage_report([], []):
            assert r.separation == 0.0

    def test_separation_is_symmetric(self):
        r = LeakageReport("x", 0.2, 0.9)
        assert r.separation == pytest.approx(0.7)


class TestRules:
    def test_every_group_compiles(self):
        assert set(rules.COMPILED) == set(rules.ALL_GROUPS)

    def test_catches_negative_parallelism(self):
        assert rules.count_group("This is not just a tool, it's a philosophy.", "negative_parallelism") >= 1

    def test_catches_not_only_but_also(self):
        assert rules.count_group("Not only fast but also cheap.", "negative_parallelism") >= 1

    def test_catches_participle_tail(self):
        text = "Revenue tripled, highlighting the strength of the model."
        assert rules.count_group(text, "participle_tail") >= 1

    def test_catches_inflated_significance(self):
        assert rules.count_group("It stands as a testament to skill.", "inflated_significance") >= 1

    def test_catches_transition_crutch(self):
        assert rules.count_group("Moreover, it was cheap. Furthermore, it worked.", "transition_crutch") == 2

    def test_case_insensitive(self):
        assert rules.count_group("MOREOVER, it failed.", "transition_crutch") >= 1

    def test_clean_prose_scores_zero_on_most_groups(self):
        text = "The pump failed on Tuesday. We replaced the seal and it ran clean."
        hits = {g: rules.count_group(text, g) for g in rules.ALL_GROUPS}
        assert sum(hits.values()) == 0, hits


class TestFeatures:
    def test_returns_every_declared_feature(self):
        f = features.extract("A sentence here. And another one there, slightly longer.")
        for name in features.FEATURE_NAMES:
            assert name in f, name

    def test_empty_text_is_safe(self):
        f = features.extract("")
        assert f["words"] == 0.0
        assert all(isinstance(v, float) for v in f.values())

    def test_whitespace_only_is_safe(self):
        assert features.extract("   \n\n  ")["words"] == 0.0

    def test_uniform_cadence_scores_low_variance(self):
        uniform = " ".join(["The cat sat on the mat today."] * 8)
        varied = "Short. " + "The cat sat on the mat and considered the long afternoon ahead of it. " * 3 + "Then it slept."
        assert features.extract(uniform)["cv_sentence_len"] < features.extract(varied)["cv_sentence_len"]

    def test_rule_features_scale_by_length(self):
        one = "Moreover, it failed. " + "Filler words here. " * 5
        many = "Moreover, it failed. " * 6
        assert features.extract(many)["rule_transition_crutch"] > features.extract(one)["rule_transition_crutch"]

    def test_triplet_detection(self):
        assert features.extract("We had eggs, bread, and milk.")["triplet_rate"] > 0

    def test_matrix_shape_and_log_scaling(self):
        rows = features.extract_many(["One sentence here.", "Another sentence there."])
        m = features.to_matrix(rows)
        assert m.shape == (2, len(features.FEATURE_NAMES))
        widx = features.FEATURE_NAMES.index("words")
        assert m[0, widx] == pytest.approx(math.log1p(rows[0]["words"]))

    def test_matrix_accepts_feature_subset(self):
        rows = features.extract_many(["Text here to measure."])
        m = features.to_matrix(rows, rules.LEXICAL_FEATURES)
        assert m.shape == (1, len(rules.LEXICAL_FEATURES))

    def test_no_nan_or_inf(self):
        for text in ["", "a", "One. Two. Three.", "x" * 5000]:
            for k, v in features.extract(text).items():
                assert not math.isnan(v) and not math.isinf(v), (text[:20], k, v)
