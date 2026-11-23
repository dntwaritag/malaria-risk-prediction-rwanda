import io

from tests.conftest import HH


def records(df, n=40):
    return df[["cluster"] + HH].head(n).to_dict("records")


def test_health(client):
    r = client.get("/health"); assert r.status_code == 200 and r.json() == {"status": "ok", "model_loaded": True}


def test_home_page_and_security_headers(client):
    r = client.get("/"); assert r.status_code == 200 and "Rwanda" in r.text
    assert r.headers["x-content-type-options"] == "nosniff" and "default-src 'self'" in r.headers["content-security-policy"]


def test_model_card_and_clusters(client):
    assert client.get("/api/model").json()["evaluation"]["pr_auc_mean"] == 0.08
    assert len(client.get("/api/clusters").json()["clusters"]) == 2
    assert [c["cluster"] for c in client.get("/api/clusters?tier=highest").json()["clusters"]] == [1]


def test_score_json(client, artefacts):
    _, df = artefacts
    j = client.post("/api/score", json={"records": records(df)}).json()
    assert j["n"] == 40 and j["scores_are_relative"] and all(0 <= r["risk_score"] <= 1 for r in j["results"])
    ranks = [c["rank"] for c in j["clusters"]]; assert ranks == sorted(ranks)


def test_missing_and_extra_fields(client):
    j = client.post("/api/score", json={"records": [{"hv105": 30, "bogus": 1}, {"hv105": 5}]}).json()
    assert j["n"] == 2 and "bogus" in j["ignored_fields"] and "hv271" in j["missing_features"]


def test_score_csv_and_bad_input(client, artefacts):
    _, df = artefacts
    buf = df[["cluster"] + HH].head(30).to_csv(index=False).encode()
    assert client.post("/api/score/csv", files={"file": ("a.csv", io.BytesIO(buf), "text/csv")}).json()["n"] == 30
    assert client.post("/api/score/csv", files={"file": ("a.csv", io.BytesIO(b"\x00\xff"), "text/csv")}).status_code in (422, 200)
    assert client.post("/api/score", json={"records": []}).status_code == 422
    assert client.post("/api/score", json={}).status_code == 422


def test_template(client):
    assert client.get("/api/template.csv").text.startswith("cluster,hv105")


def test_docs_not_blocked_by_csp(client):
    r = client.get("/docs"); assert r.status_code == 200 and "content-security-policy" not in r.headers
