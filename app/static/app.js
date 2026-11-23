(function () {
  "use strict";
  var COLORS = {Highest: "#c41230", High: "#e8a33d", Medium: "#1b998b", Lower: "#9aa8bd"};
  function el(tag, text) { var e = document.createElement(tag); if (text !== undefined) e.textContent = text; return e; }
  function get(url) { return fetch(url).then(function (r) { if (!r.ok) throw new Error(url + " " + r.status); return r.json(); }); }
  function fill(id, v) { document.getElementById(id).textContent = v; }

  get("/api/model").then(function (m) {
    var e = m.evaluation;
    fill("s-pr", e.pr_auc_mean.toFixed(3)); fill("s-base", e.baseline_pr_auc.toFixed(4));
    fill("s-roc", e.roc_auc_mean.toFixed(3)); fill("s-lift", e.lift_top20.toFixed(1) + "x");
    fill("s-n", m.training_data.n.toLocaleString()); fill("s-pos", m.training_data.positives);
    fill("about", m.model + ". Evaluation: " + e.protocol + ". PR-AUC " + e.pr_auc_mean.toFixed(3) + " ± " + e.pr_auc_sd.toFixed(3) + " across repeats.");
    var ul = document.getElementById("limits"); m.limitations.forEach(function (t) { ul.appendChild(el("li", t)); });
  }).catch(function () { fill("about", "Model information is not available."); });

  var rows = [];
  function draw() {
    var tier = document.getElementById("tier").value;
    var data = rows.filter(function (r) { return !tier || r.tier === tier; });
    var tb = document.querySelector("#clusters tbody"); tb.textContent = "";
    data.forEach(function (r) {
      var tr = el("tr"); [r.rank, r.cluster, r.risk_score.toFixed(4), r.tier].forEach(function (v) { tr.appendChild(el("td", String(v))); }); tb.appendChild(tr);
    });
    var W = 960, H = 220, pad = 28, max = Math.max.apply(null, data.map(function (r) { return r.risk_score; }).concat([0.001]));
    var bw = data.length ? (W - pad) / data.length : 0, ns = "http://www.w3.org/2000/svg";
    var svg = document.createElementNS(ns, "svg"); svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    data.forEach(function (r, i) {
      var h = (r.risk_score / max) * (H - pad - 6), b = document.createElementNS(ns, "rect");
      b.setAttribute("x", pad + i * bw); b.setAttribute("y", H - pad - h); b.setAttribute("width", Math.max(bw - 1, 1)); b.setAttribute("height", h);
      b.setAttribute("fill", COLORS[r.tier] || "#999"); var t = document.createElementNS(ns, "title"); t.textContent = "Cluster " + r.cluster + ": " + r.risk_score.toFixed(4); b.appendChild(t); svg.appendChild(b);
    });
    var lab = document.createElementNS(ns, "text"); lab.setAttribute("x", pad); lab.setAttribute("y", H - 8); lab.setAttribute("font-size", "12"); lab.setAttribute("fill", "#5a6678");
    lab.textContent = "Clusters, highest risk first (" + data.length + ")"; svg.appendChild(lab);
    var c = document.getElementById("chart"); c.textContent = ""; c.appendChild(svg);
  }
  get("/api/clusters").then(function (d) { rows = d.clusters; draw(); }).catch(function () { fill("msg", ""); });
  document.getElementById("tier").addEventListener("change", draw);

  document.getElementById("go").addEventListener("click", function () {
    var f = document.getElementById("file").files[0], msg = document.getElementById("msg"), tbl = document.getElementById("results");
    if (!f) { msg.textContent = "Choose a CSV file first."; return; }
    var fd = new FormData(); fd.append("file", f); msg.textContent = "Scoring...";
    fetch("/api/score/csv", {method: "POST", body: fd}).then(function (r) { return r.json().then(function (j) { if (!r.ok) throw new Error(j.detail || r.status); return j; }); })
      .then(function (j) {
        var tb = tbl.querySelector("tbody"); tb.textContent = "";
        j.clusters.forEach(function (c) { var tr = el("tr"); [c.rank, c.cluster, c.n, c.mean_risk.toFixed(4)].forEach(function (v) { tr.appendChild(el("td", String(v))); }); tb.appendChild(tr); });
        tbl.hidden = false;
        msg.textContent = "Scored " + j.n + " records." + (j.missing_features.length ? " Missing columns treated as unknown: " + j.missing_features.length + "." : "");
      }).catch(function (e) { msg.textContent = "Could not score the file: " + e.message; tbl.hidden = true; });
  });
})();
