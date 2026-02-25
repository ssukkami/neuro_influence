
function switchTab(tabId) {
    document.querySelectorAll('.view').forEach(v => {
        v.classList.remove('active-view');
        v.style.display = 'none';
    });
    const active = document.getElementById(tabId);
    if (active) { active.classList.add('active-view'); active.style.display = ''; }

    document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
    document.querySelector(`.nav-item[data-tab="${tabId}"]`)?.classList.add('active');
}

document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.nav-item').forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });
    switchTab('home');
    document.getElementById('btn-run')?.addEventListener('click', runAnalysis);
    initABTesting();
});

async function runAnalysis() {
    const btn  = document.getElementById('btn-run');
    const text = (document.getElementById('input-text')?.value || '').trim();
    if (!text) return alert('INPUT REQUIRED');

    const payload = {
        text,
        followers:    +document.getElementById('input-followers').value  || 1000,
        following:    +document.getElementById('input-following').value  || 500,
        num_posts:    +document.getElementById('input-posts').value      || 100,
        avg_likes:    +document.getElementById('input-avg-likes').value  || 0,
        account_type:  document.getElementById('input-type').value       || 'CREATOR',
    };

    const orig = btn.innerHTML;
    btn.innerHTML = 'PROCESSING... <i class="fa-solid fa-circle-notch fa-spin"></i>';
    btn.disabled = true;

    try {
        const res  = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (!res.ok || data.error) return alert('MODEL ERROR: ' + (data.error || 'UNKNOWN'));
        renderResults(data);
        setText('result-status', '[ COMPLETE ]');
    } catch {
        alert('CONNECTION FAILED');
    } finally {
        btn.innerHTML = orig;
        btn.disabled  = false;
    }
}

function renderResults(data) {
    const score = Number(data.engagement_score    ?? 0);
    const pred  = Number(data.predicted_engagement ?? 0);

    animateVal('out-score', 0, score, 800);
    setText('score-sublabel', `EXPECTED INTERACTIONS: ${formatInt(pred)}`);

    const pos = Number(data.sentiment_pos ?? 0) * 100;
    const neu = Number(data.sentiment_neu ?? 0) * 100;
    const neg = Number(data.sentiment_neg ?? 0) * 100;
    setText('out-pos', `${pos.toFixed(1)}%`);
    setText('out-neu', `${neu.toFixed(1)}%`);
    setText('out-neg', `${neg.toFixed(1)}%`);
    setWidth('bar-pos', pos);
    setWidth('bar-neu', neu);
    setWidth('bar-neg', neg);

    const deltaEl = document.getElementById('out-delta');
    if (deltaEl) {
        if (data.delta_ok && data.growth_percent_vs_avg != null) {
            const g = Number(data.growth_percent_vs_avg);
            deltaEl.textContent = `${g >= 0 ? '+' : ''}${g.toFixed(1)}%`;
            deltaEl.style.color = g >= 0 ? '#fff' : 'var(--accent)';
        } else { deltaEl.textContent = '—'; deltaEl.style.color = ''; }
    }

    renderGauge('gauge-lang', Number(data.language_score     ?? 0), '#ffffff');
    renderGauge('gauge-aud',  Number(data.audience_fit_score ?? 0), '#ff3300');

    const s  = data.signals || {};
    const el = document.getElementById('signals-body');
    if (el) {
        const chips = [
            { lbl:'CTA',       val: s.cta_detected ? 'YES' : 'NO', ok: s.cta_detected, cta: true },
            { lbl:'HASHTAGS',  val: s.hashtag_count  ?? 0 },
            { lbl:'QUESTIONS', val: s.question_count ?? 0 },
            { lbl:'MENTIONS',  val: s.mention_count  ?? 0 },
            { lbl:'EMOJIS',    val: s.emoji_count    ?? 0 },
        ];
        el.innerHTML = chips.map(c => `
          <div class="signal-chip ${c.cta ? (c.ok ? 'cta-yes':'cta-no') : ''}">
            <span class="chip-val">${c.val}</span>
            <span class="chip-lbl">${c.lbl}</span>
          </div>`).join('');
    }

    const diags  = Array.isArray(data.diagnostics)     ? data.diagnostics     : [];
    const recs   = Array.isArray(data.recommendations) ? data.recommendations : [];
    const phrases= Array.isArray(data.top_phrases)     ? data.top_phrases     : [];

    const diagEl = document.getElementById('diag-body');
    if (diagEl) diagEl.innerHTML = diags.length
        ? diags.map(x=>`<div class="list-item"><span>→</span><span>${escHtml(x)}</span></div>`).join('')
        : '<div style="opacity:.4;font-size:.75rem;">—</div>';

    const recEl = document.getElementById('recs-body');
    if (recEl) recEl.innerHTML = recs.length
        ? recs.map((x,i)=>`<div class="list-item"><span class="list-num">${String(i+1).padStart(2,'0')}</span><span>${escHtml(x)}</span></div>`).join('')
        : '<div style="opacity:.4;font-size:.75rem;">—</div>';

    const phEl = document.getElementById('phrases-body');
    if (phEl) phEl.innerHTML = phrases.length
        ? phrases.map(p=>`<div class="phrase-chip"><span>${escHtml(p.phrase)}</span><span class="ph-score">${p.score}</span></div>`).join('')
        : '<div style="opacity:.4;font-size:.75rem;">—</div>';
}

function renderGauge(elId, value, color) {
    const el = document.getElementById(elId);
    if (!el) return;
    const pct   = clamp(value, 0, 100);
    const grade = pct >= 75 ? 'STRONG' : pct >= 50 ? 'FAIR' : 'WEAK';
    el.innerHTML = `
      <div class="gauge-bar-wrap">
        <div class="gauge-track">
          <div class="gauge-fill" style="width:${pct}%;background:${color};"></div>
        </div>
      </div>
      <div class="gauge-meta">
        <span style="color:${color};">${pct.toFixed(0)} / 100</span>
        <span style="color:#555;">${grade}</span>
      </div>`;
}

function initABTesting() {
    document.getElementById('btn-compare')?.addEventListener('click', async () => {
        const btn   = document.getElementById('btn-compare');
        const textA = (document.getElementById('ab-a')?.value || '').trim();
        const textB = (document.getElementById('ab-b')?.value || '').trim();
        if (!textA || !textB) return alert('INPUT BOTH VARIANTS');

        btn.innerHTML = 'SIMULATING... <i class="fa-solid fa-circle-notch fa-spin"></i>';
        btn.disabled  = true;

        const base = { followers:1000, following:500, num_posts:100, account_type:'CREATOR', avg_likes:0 };
        const analyze = async (txt) => {
            const r = await fetch('/api/analyze', {
                method:'POST', headers:{'Content-Type':'application/json'},
                body: JSON.stringify({ text:txt, ...base }),
            });
            const j = await r.json();
            if (!r.ok || j.error) throw new Error(j.error || 'MODEL_ERROR');
            return j;
        };

        try {
            const [a, b] = await Promise.all([analyze(textA), analyze(textB)]);
            renderABCard('ab-a-res', a, b);
            renderABCard('ab-b-res', b, a);
            const diff = Number(b.engagement_score??0) - Number(a.engagement_score??0);
            if (Math.abs(diff)<0.5) btn.innerHTML = 'NO SIGNIFICANT DIFFERENCE';
            else if (diff>0)        btn.innerHTML = `VARIANT B WINS  (+${diff.toFixed(1)})`;
            else                    btn.innerHTML = `VARIANT A WINS  (+${Math.abs(diff).toFixed(1)})`;
        } catch(e) {
            btn.innerHTML = 'FAILED: ' + e.message;
        } finally { btn.disabled = false; }
    });
}

function renderABCard(elId, own, other) {
    const el = document.getElementById(elId);
    if (!el) return;
    const score = Number(own.engagement_score   ?? 0);
    const diff  = score - Number(other.engagement_score ?? 0);
    const color = diff >= 0 ? '#fff' : 'var(--accent)';
    const sign  = diff >= 0 ? '+' : '';
    const lang  = Number(own.language_score     ?? 0).toFixed(0);
    const aud   = Number(own.audience_fit_score ?? 0).toFixed(0);
    const s     = own.signals || {};
    const recs  = (own.recommendations || []).slice(0,2);

    el.innerHTML = `
      <div style="font-size:1.8rem;font-weight:700;font-family:'Space Grotesk',sans-serif;color:${color};">
        ${score.toFixed(1)}
      </div>
      <div style="font-size:.62rem;color:#555;margin:.2rem 0 .4rem;">
        SCORE &nbsp;|&nbsp; ${sign}${diff.toFixed(1)} vs other
      </div>
      <div style="font-size:.68rem;color:#888;margin-bottom:.4rem;">
        LANG:<span style="color:#fff;"> ${lang}</span>
        &nbsp;AUD:<span style="color:var(--accent);"> ${aud}</span>
        &nbsp;CTA:<span style="color:${s.cta_detected?'#fff':'var(--accent)'}"> ${s.cta_detected?'YES':'NO'}</span>
        &nbsp;#:<span style="color:#fff;"> ${s.hashtag_count??0}</span>
        &nbsp;😊:<span style="color:#fff;"> ${s.emoji_count??0}</span>
      </div>
      ${recs.map(r=>`<div style="font-size:.65rem;color:#666;border-top:1px solid #1a1a1a;padding:.18rem 0;">→ ${escHtml(r)}</div>`).join('')}
    `;
}

function animateVal(id, from, to, ms) {
    const el = document.getElementById(id); if (!el) return;
    let t0 = null;
    const step = ts => {
        if (!t0) t0 = ts;
        const p = Math.min((ts-t0)/ms, 1);
        el.textContent = (p*(to-from)+from).toFixed(1);
        if (p<1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
}
function setText(id,v)    { const e=document.getElementById(id); if(e) e.textContent=v; }
function setWidth(id,pct) { const e=document.getElementById(id); if(e) e.style.width=`${clamp(pct,0,100)}%`; }
function clamp(x,a,b)     { return Math.max(a,Math.min(b,x)); }
function formatInt(x)     { return Math.round(Number(x)||0).toString(); }
function escHtml(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
        .replace(/"/g,'&quot;').replace(/'/g,'&#039;');
}