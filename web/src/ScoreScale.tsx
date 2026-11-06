// The result as a measuring instrument: a 0-1 ruler, the threshold tick, and the score marker.
import type { Prediction } from "./api";

export function ScoreScale({ result }: { result: Prediction }) {
  const pct = (v: number) => `${Math.min(100, Math.max(0, v * 100))}%`;
  return (
    <figure className={`scale ${result.flagged ? "is-flagged" : ""}`}>
      <figcaption className="scale-reading">
        <span className="scale-score">{result.score.toFixed(2)}</span>
        <span className="scale-verdict">
          {result.flagged
            ? "Above the threshold: this text resembles r/SuicideWatch posts."
            : "Below the threshold: this text does not resemble r/SuicideWatch posts enough to be flagged."}
        </span>
      </figcaption>

      <div className="ruler" role="img"
           aria-label={`Score ${result.score.toFixed(2)} on a scale from 0 to 1; threshold ${result.threshold.toFixed(2)}`}>
        <div className="ruler-track" />
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <span key={t} className="ruler-tick" style={{ left: pct(t) }}><span>{t}</span></span>
        ))}
        {/* near the right edge, the label goes on the left of the line so it never overflows */}
        <span className={`ruler-threshold ${result.threshold > 0.7 ? "label-left" : ""}`} style={{ left: pct(result.threshold) }}>
          <span>threshold {result.threshold.toFixed(2)}</span>
        </span>
        <span className="ruler-marker" style={{ left: pct(result.score) }} />
      </div>

      <p className="scale-note">
        This is a similarity score, not an assessment of the writer. The threshold was set on validation data so the
        model finds 90 % of r/SuicideWatch posts; it does not transfer well to short sentences.
      </p>
      {/* Shown for EVERY result: the model misses real distress (false negatives), so help must never depend on the score */}
      <p className="scale-help">
        A low score does not mean someone is fine: this model misses many real messages of distress. If this text is
        about you or someone you know, <a href={result.help_url} target="_blank" rel="noreferrer">find a helpline near you</a>.
      </p>
    </figure>
  );
}
