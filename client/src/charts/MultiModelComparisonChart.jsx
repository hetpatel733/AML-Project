import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

/**
 * MultiModelComparisonChart
 * D3.js visualization comparing candidate models and ensemble predictions for the specific analyzed article.
 */
export const MultiModelComparisonChart = ({ models = [], ensembles = null, height = 320 }) => {
  const containerRef = useRef(null);
  const svgRef = useRef(null);
  const tooltipRef = useRef(null);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return;

    d3.select(svgRef.current).selectAll('*').remove();

    const containerWidth = containerRef.current.clientWidth || 600;
    const margin = { top: 25, right: 30, bottom: 65, left: 60 };
    const width = containerWidth - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    const svg = d3.select(svgRef.current)
      .attr('width', containerWidth)
      .attr('height', height)
      .attr('viewBox', `0 0 ${containerWidth} ${height}`)
      .append('g')
      .attr('transform', `translate(${margin.left}, ${margin.top})`);

    // Prepare combined data list: 4 trained models + 2 ensembles
    const items = [];
    
    if (models && models.length > 0) {
      models.forEach(m => {
        items.push({
          id: m.model_id,
          name: m.model_name.replace('Multinomial Naive Bayes', 'MNB').replace('Passive Aggressive', 'PAC').replace('Logistic Regression', 'LogReg'),
          fullName: m.model_name,
          realProb: m.probabilities?.REAL ?? (m.prediction === 'REAL' ? m.confidence : 1 - m.confidence),
          fakeProb: m.probabilities?.FAKE ?? (m.prediction === 'FAKE' ? m.confidence : 1 - m.confidence),
          prediction: m.prediction,
          confidence: m.confidence,
          type: 'single'
        });
      });
    }

    if (ensembles?.weighted_soft_ensemble) {
      const soft = ensembles.weighted_soft_ensemble;
      items.push({
        id: 'soft_ens',
        name: 'Soft Ensemble',
        fullName: 'Validation-Weighted Soft Ensemble',
        realProb: soft.probabilities?.REAL ?? 0.5,
        fakeProb: soft.probabilities?.FAKE ?? 0.5,
        prediction: soft.prediction,
        confidence: soft.confidence,
        type: 'ensemble'
      });
    }

    if (ensembles?.majority_voting_hard_ensemble) {
      const hard = ensembles.majority_voting_hard_ensemble;
      const isReal = hard.prediction === 'REAL';
      items.push({
        id: 'hard_ens',
        name: 'Hard Ensemble',
        fullName: 'Majority Voting Hard Ensemble (Vote %)',
        realProb: isReal ? (hard.vote_percentage / 100) : (1 - hard.vote_percentage / 100),
        fakeProb: isReal ? (1 - hard.vote_percentage / 100) : (hard.vote_percentage / 100),
        prediction: hard.prediction,
        confidence: hard.confidence,
        type: 'ensemble'
      });
    }

    if (items.length === 0) return;

    // X Scale
    const x = d3.scaleBand()
      .domain(items.map(d => d.name))
      .range([0, width])
      .padding(0.3);

    // Y Scale (0% to 100% probability)
    const y = d3.scaleLinear()
      .domain([0, 1.0])
      .range([chartHeight, 0]);

    // Grid Lines
    svg.append('g')
      .attr('class', 'grid-lines')
      .call(
        d3.axisLeft(y)
          .tickSize(-width)
          .tickFormat('')
          .ticks(5)
      )
      .selectAll('line')
      .attr('stroke', 'rgba(255, 255, 255, 0.05)')
      .attr('stroke-dasharray', '3,3');

    // 50% Threshold Baseline
    svg.append('line')
      .attr('x1', 0)
      .attr('x2', width)
      .attr('y1', y(0.5))
      .attr('y2', y(0.5))
      .attr('stroke', '#64748b')
      .attr('stroke-dasharray', '4,4')
      .attr('stroke-width', 1.5)
      .attr('opacity', 0.6);

    svg.append('text')
      .attr('x', width - 5)
      .attr('y', y(0.5) - 6)
      .attr('text-anchor', 'end')
      .attr('fill', '#94a3b8')
      .attr('font-size', '10px')
      .attr('font-family', 'Inter, sans-serif')
      .text('50% Decision Threshold');

    const tooltip = d3.select(tooltipRef.current);

    // Render Bars
    svg.selectAll('.bar-group')
      .data(items)
      .enter()
      .append('rect')
      .attr('class', 'model-bar')
      .attr('x', d => x(d.name))
      .attr('y', d => y(d.prediction === 'REAL' ? d.realProb : d.fakeProb))
      .attr('width', x.bandwidth())
      .attr('height', d => Math.max(4, chartHeight - y(d.prediction === 'REAL' ? d.realProb : d.fakeProb)))
      .attr('rx', 4)
      .attr('fill', d => {
        if (d.type === 'ensemble') {
          return d.prediction === 'REAL' ? '#8b5cf6' : '#ec4899';
        }
        return d.prediction === 'REAL' ? '#10b981' : '#ef4444';
      })
      .style('cursor', 'pointer')
      .on('mouseenter', function (event, d) {
        const barColor = d.prediction === 'REAL' ? '#10b981' : '#ef4444';
        d3.select(this).attr('filter', `drop-shadow(0 0 8px ${barColor}90)`);

        tooltip
          .style('opacity', 1)
          .html(`
            <div style="font-weight: 700; color: #f8fafc; margin-bottom: 4px;">${d.fullName}</div>
            <div style="color: ${d.prediction === 'REAL' ? '#10b981' : '#ef4444'}; font-weight: 600;">
              Verdict: ${d.prediction} (${((d.prediction === 'REAL' ? d.realProb : d.fakeProb) * 100).toFixed(1)}%)
            </div>
            <div style="font-size: 11px; color: #94a3b8; margin-top: 4px;">
              P(REAL): ${(d.realProb * 100).toFixed(1)}% | P(FAKE): ${(d.fakeProb * 100).toFixed(1)}%
            </div>
          `);
      })
      .on('mousemove', function (event) {
        const bounds = containerRef.current.getBoundingClientRect();
        tooltip
          .style('left', `${event.clientX - bounds.left + 15}px`)
          .style('top', `${event.clientY - bounds.top - 10}px`);
      })
      .on('mouseleave', function () {
        d3.select(this).attr('filter', null);
        tooltip.style('opacity', 0);
      });

    // Add Value Labels on top of bars
    svg.selectAll('.bar-label')
      .data(items)
      .enter()
      .append('text')
      .attr('x', d => (x(d.name) || 0) + x.bandwidth() / 2)
      .attr('y', d => y(d.prediction === 'REAL' ? d.realProb : d.fakeProb) - 6)
      .attr('text-anchor', 'middle')
      .attr('fill', '#f1f5f9')
      .attr('font-size', '11px')
      .attr('font-weight', '600')
      .text(d => `${((d.prediction === 'REAL' ? d.realProb : d.fakeProb) * 100).toFixed(0)}%`);

    // X Axis
    svg.append('g')
      .attr('transform', `translate(0, ${chartHeight})`)
      .call(d3.axisBottom(x))
      .selectAll('text')
      .attr('transform', 'rotate(-25)')
      .style('text-anchor', 'end')
      .attr('dx', '-0.5em')
      .attr('dy', '0.7em')
      .attr('fill', '#94a3b8')
      .attr('font-size', '11px');

    // Y Axis
    svg.append('g')
      .call(
        d3.axisLeft(y)
          .ticks(5)
          .tickFormat(d => `${(d * 100).toFixed(0)}%`)
      )
      .selectAll('text')
      .attr('fill', '#94a3b8')
      .attr('font-size', '11px');

  }, [models, ensembles, height]);

  return (
    <div className="chart-wrapper" ref={containerRef} style={{ position: 'relative', width: '100%' }}>
      <svg ref={svgRef} style={{ width: '100%', overflow: 'visible' }} />
      <div
        ref={tooltipRef}
        className="d3-chart-tooltip"
        style={{
          position: 'absolute',
          opacity: 0,
          pointerEvents: 'none',
          backgroundColor: 'rgba(15, 23, 42, 0.95)',
          border: '1px solid rgba(255, 255, 255, 0.15)',
          borderRadius: '8px',
          padding: '8px 12px',
          color: '#fff',
          fontSize: '12px',
          zIndex: 50,
          boxShadow: '0 10px 25px rgba(0,0,0,0.5)',
          backdropFilter: 'blur(8px)',
          transition: 'opacity 0.15s ease'
        }}
      />
    </div>
  );
};

export default MultiModelComparisonChart;
