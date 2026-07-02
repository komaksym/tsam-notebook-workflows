"""Offline, client-rendered feature drilldowns for grouped TSAM results."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd
from plotly.utils import PlotlyJSONEncoder

from tsam_workflows.config import country_options


def _column_values(frame: pd.DataFrame) -> dict[str, list[Any]]:
    """Return column-oriented values suitable for compact JSON serialization."""
    return {str(column): frame[column].tolist() for column in frame.columns}


def _representative_values(result: Any) -> dict[str, dict[str, list[Any]]]:
    """Return representative profiles keyed by column and cluster ID."""
    representatives = result.cluster_representatives
    cluster_ids = representatives.index.get_level_values(0).unique()
    return {
        str(column): {
            str(cluster_id): representatives.loc[cluster_id, column].tolist()
            for cluster_id in cluster_ids
        }
        for column in representatives.columns
    }


def build_drilldown_payload(result: Any) -> dict[str, Any]:
    """Serialize reusable TSAM arrays once per group for browser-side charts."""
    groups: dict[str, Any] = {}
    group_ids = getattr(result, "group_ids", result.tsam_results_by_group)
    for group_id in group_ids:
        aggregation = result.tsam_results_by_group[group_id]
        original = aggregation.original
        groups[group_id] = {
            "timestamps": [timestamp.isoformat() for timestamp in original.index],
            "original": _column_values(original),
            "representatives": _representative_values(aggregation),
            "assignments": list(aggregation.cluster_assignments),
            "weights": {
                str(cluster_id): weight
                for cluster_id, weight in aggregation.cluster_weights.items()
            },
            "timesteps": aggregation.n_timesteps_per_period,
        }
    options = country_options(list(result.feature_columns_by_country_and_group))
    return {
        "groups": groups,
        "features": result.feature_columns_by_country_and_group,
        "country_labels": {code: label for label, code in options},
        "representative_mode": (
            "Mean-preserved synthetic representatives"
            if getattr(result, "preserve_column_means", False)
            else "Observed medoid representatives"
        ),
    }


def _option(value: str, label: str) -> str:
    """Return one escaped HTML select option."""
    return (
        f'<option value="{html.escape(value, quote=True)}">'
        f"{html.escape(label)}</option>"
    )


def write_drilldown_dashboard(result: Any, output_dir: Path) -> Path:
    """Write one offline dashboard that renders selected drilldowns on demand."""
    payload = json.dumps(
        build_drilldown_payload(result),
        cls=PlotlyJSONEncoder,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    groups = getattr(result, "group_ids", result.tsam_results_by_group)
    group_options = "".join(_option(group, group) for group in groups)
    chart_options = "".join(
        _option(value, label)
        for value, label in (
            ("representatives", "Cluster representatives"),
            ("members", "Cluster members"),
            ("comparison", "Original vs reconstructed"),
            ("residuals", "Residuals"),
        )
    )
    representative_mode = (
        "Mean-preserved synthetic representatives"
        if getattr(result, "preserve_column_means", False)
        else "Observed medoid representatives"
    )
    path = output_dir / "drilldown_dashboard.html"
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>TSAM feature drilldowns</title>"
        "<style>"
        "body{font-family:system-ui,sans-serif;margin:24px;color:#2a3f5f;}"
        ".controls{display:flex;flex-wrap:wrap;gap:16px;margin-bottom:16px;}"
        "label{display:grid;gap:4px;font-weight:600;}"
        "select{font:inherit;min-width:190px;padding:7px;}"
        "#chart{min-height:720px;}[hidden]{display:none!important;}"
        "</style></head><body>"
        "<h1>TSAM feature drilldowns</h1>"
        f'<p id="representative-mode">{html.escape(representative_mode)}</p>'
        '<div class="controls">'
        '<label>Group<select id="group-select">'
        f"{group_options}</select></label>"
        '<label>Country<select id="country-select"></select></label>'
        '<label>Feature group<select id="feature-select"></select></label>'
        '<label>Chart<select id="chart-select">'
        f"{chart_options}</select></label>"
        '<label id="cluster-control" hidden>Cluster'
        '<select id="cluster-select"></select></label>'
        "</div>"
        '<div id="chart"></div>'
        '<script src="plotly.min.js"></script><script>'
        f"const payload={payload};"
        "const groupSelect=document.getElementById('group-select');"
        "const countrySelect=document.getElementById('country-select');"
        "const featureSelect=document.getElementById('feature-select');"
        "const chartSelect=document.getElementById('chart-select');"
        "const clusterSelect=document.getElementById('cluster-select');"
        "const clusterControl=document.getElementById('cluster-control');"
        "const config={responsive:true};"
        "const colors=['#636efa','#EF553B','#00cc96','#ab63fa','#FFA15A',"
        "'#19d3f3','#FF6692','#B6E880','#FF97FF','#FECB52'];"
        "function setOptions(select,values,labels={}){"
        "const previous=select.value;select.replaceChildren(...values.map(value=>{"
        "const option=document.createElement('option');option.value=value;"
        "option.textContent=labels[value]??value;return option;}));"
        "if(values.includes(previous)){select.value=previous;}"
        "}"
        "function updateCountries(){"
        "const countries=Object.keys(payload.features).sort();"
        "setOptions(countrySelect,countries,payload.country_labels);"
        "if(countries.includes('DE')){countrySelect.value='DE';}"
        "updateFeatures();}"
        "function updateFeatures(){"
        "setOptions(featureSelect,Object.keys(payload.features[countrySelect.value]).sort());"
        "render();}"
        "function columns(){return payload.features[countrySelect.value][featureSelect.value];}"
        "function group(){return payload.groups[groupSelect.value];}"
        "function clusterIds(){return Object.keys(group().weights).sort((a,b)=>Number(a)-Number(b));}"
        "function layout(title){return {"
        "title:{text:title,x:0.5,xanchor:'center',font:{size:20}},"
        "autosize:true,height:720,margin:{l:60,r:30,t:100,b:70},"
        "paper_bgcolor:'white',plot_bgcolor:'rgb(237,237,237)',"
        "font:{color:'rgb(51,51,51)'},hovermode:'closest',"
        "xaxis:{gridcolor:'white',zerolinecolor:'white'},"
        "yaxis:{gridcolor:'white',zerolinecolor:'white'},"
        "legend:{yanchor:'top',y:1,xanchor:'left',x:1.02}};}"
        "function buildRepresentatives(){"
        "const selected=group();const traces=[];"
        "columns().forEach((column,columnIndex)=>clusterIds().forEach(cluster=>{"
        "const values=selected.representatives[column][cluster];"
        "traces.push({type:'scatter',mode:'lines',x:values.map((_,i)=>i),y:values,"
        "name:`${column} — Period ${cluster} (n=${selected.weights[cluster]})`,"
        "line:{color:colors[(Number(cluster)+columnIndex)%colors.length]}});}));"
        "return {traces,layout:layout(`Cluster representative profiles: ${groupSelect.value}`)};}"
        "function updateClusters(){setOptions(clusterSelect,clusterIds());}"
        "function buildMembers(){"
        "const selected=group();const cluster=Number(clusterSelect.value);"
        "const n=selected.timesteps;const traces=[];"
        "columns().forEach((column,columnIndex)=>{const x=[];const y=[];"
        "selected.assignments.forEach((assignment,period)=>{if(assignment!==cluster)return;"
        "const start=period*n;for(let step=0;step<n;step++){x.push(step);"
        "y.push(selected.original[column][start+step]);}x.push(null);y.push(null);});"
        "traces.push({type:'scatter',mode:'lines',x,y,name:`${column} members`,"
        "line:{color:'rgba(99,110,250,0.3)'},connectgaps:false});"
        "const representative=selected.representatives[column][String(cluster)];"
        "traces.push({type:'scatter',mode:'lines',x:representative.map((_,i)=>i),"
        "y:representative,name:`${column} representative`,"
        "line:{color:colors[(columnIndex+1)%colors.length],width:3}});});"
        "return {traces,layout:layout(`Cluster members: ${groupSelect.value}, cluster ${cluster}`)};}"
        "function buildComparison(){"
        "const selected=group();const traces=[];columns().forEach((column,index)=>{"
        "const reconstructed=reconstruct(selected,column);"
        "traces.push({type:'scatter',mode:'lines',x:selected.timestamps,"
        "y:selected.original[column],name:`${column} — Original`,"
        "line:{color:colors[index%colors.length]}});"
        "traces.push({type:'scatter',mode:'lines',x:selected.timestamps,"
        "y:reconstructed,name:`${column} — Reconstructed`,"
        "line:{color:colors[index%colors.length],dash:'dash'}});});"
        "return {traces,layout:layout(`Original vs reconstructed: ${groupSelect.value}`)};}"
        "function reconstruct(selected,column){const values=[];"
        "selected.assignments.forEach(cluster=>values.push("
        "...selected.representatives[column][String(cluster)]));"
        "return values.slice(0,selected.original[column].length);}"
        "function buildResiduals(){"
        "const selected=group();const traces=columns().map((column,index)=>{"
        "const reconstructed=reconstruct(selected,column);return ({"
        "type:'scatter',mode:'lines',x:selected.timestamps,"
        "y:selected.original[column].map((value,i)=>value-reconstructed[i]),"
        "name:column,line:{color:colors[index%colors.length]}});});"
        "const chartLayout=layout(`Residuals: ${groupSelect.value}`);"
        "chartLayout.shapes=[{type:'line',xref:'paper',x0:0,x1:1,y0:0,y1:0,"
        "line:{color:'gray',dash:'dash'}}];return {traces,layout:chartLayout};}"
        "function render(){if(!countrySelect.value||!featureSelect.value)return;"
        "const isMembers=chartSelect.value==='members';clusterControl.hidden=!isMembers;"
        "if(isMembers&&!clusterSelect.value){updateClusters();}"
        "const builders={representatives:buildRepresentatives,members:buildMembers,"
        "comparison:buildComparison,residuals:buildResiduals};"
        "const figure=builders[chartSelect.value]();"
        "Plotly.react('chart',figure.traces,figure.layout,config);}"
        "groupSelect.addEventListener('change',()=>{updateClusters();render();});"
        "countrySelect.addEventListener('change',updateFeatures);"
        "featureSelect.addEventListener('change',render);"
        "chartSelect.addEventListener('change',render);"
        "clusterSelect.addEventListener('change',render);"
        "updateCountries();updateClusters();render();"
        "</script></body></html>",
        encoding="utf-8",
    )
    return path
