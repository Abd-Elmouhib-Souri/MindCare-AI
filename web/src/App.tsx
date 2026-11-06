// MindCare AI demo page: one text box, one honest result, and what the model gets wrong.
import { useEffect, useState } from "react";
import { ApiError, getModelCard, predict, type ModelCard, type Prediction } from "./api";
import { ScoreScale } from "./ScoreScale";
import { ChallengeStrip } from "./ChallengeStrip";
import { LiveLab } from "./LiveLab";

// Synthetic sentences from the frozen challenge set (eval/challenge/challenge_v1.csv), never real posts
const EXAMPLES: { tests: string; text: string }[] = [
  { tests: "Distress without keywords", text: "I don't see the point of waking up anymore. Every day feels heavy and the same." },
  { tests: "Distress in a teen context", text: "Everyone at school has friends on Discord. I just sit in my room and feel like nobody would notice if I disappeared." },
  { tests: "A keyword without distress", text: "Our team watched a documentary about suicide prevention at work today. Really important topic." },
  { tests: "A negation", text: "I'm not sad at all, just bored waiting for the weekend." },
];

const REPO_URL = "https://github.com/Abd-Elmouhib-Souri/Mental-Health-AI-Detection";

export default function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(false);
  const [card, setCard] = useState<ModelCard | null>(null);

  // The measured numbers come from the API (read from committed result files), not from this file
  useEffect(() => {
    getModelCard().then(setCard).catch(() => setCard(null));
  }, []);

  async function analyse() {
    setError(null);
    if (!text.trim()) {
      setError(new ApiError("invalid", "Type a sentence first."));
      return;
    }
    setLoading(true);
    try {
      setResult(await predict(text));
    } catch (e) {
      setResult(null);
      setError(e instanceof ApiError ? e : new ApiError("server", "Something unexpected happened."));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <p className="notice" role="note">
        Research prototype. It is not a diagnostic or crisis tool and must not be used to judge a person.
        If you are struggling, <a href="https://findahelpline.com" target="_blank" rel="noreferrer">find a helpline near you</a>.
      </p>

      <header className="intro">
        <h1>What does a distress-language model actually measure?</h1>
        <p>
          This model was trained on Reddit posts from r/SuicideWatch and r/teenagers. Its score tells you how much
          a text looks like the first group rather than the second. Try a sentence and see where it lands.
        </p>
      </header>

      <section className="demo" aria-labelledby="demo-title">
        <h2 id="demo-title" className="visually-hidden">Try the model</h2>
        <label htmlFor="input" className="field-label">Your text, in English</label>
        <textarea
          id="input"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) analyse(); }}
          rows={4}
          maxLength={5000}
          placeholder="Write a sentence or two…"
        />
        <div className="actions">
          <button type="button" onClick={analyse} disabled={loading}>
            {loading ? "Analysing…" : "Analyse text"}
          </button>
          <span className="privacy">Sent to your local API only. Nothing is stored.</span>
        </div>

        <div className="examples">
          <p className="examples-title">Or try a sentence from the test set:</p>
          <ul>
            {EXAMPLES.map((ex) => (
              <li key={ex.tests}>
                <button type="button" className="example" onClick={() => { setText(ex.text); setResult(null); setError(null); }}>
                  <span className="example-kind">{ex.tests}</span>
                  <span className="example-text">{ex.text}</span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      </section>

      <section className="result" aria-live="polite">
        {error && (
          <div className="error" role="alert">
            <p>{error.message}</p>
            {error.kind === "offline" && (
              <p>Start it from the repository folder with <code>uvicorn mindcare.api:app --port 8000</code>, then try again.</p>
            )}
          </div>
        )}
        {result && <ScoreScale result={result} />}
      </section>

      {card && (
        <section className="evidence" aria-labelledby="evidence-title">
          <h2 id="evidence-title">Where it fails</h2>
          <p>
            On {card.test_n.toLocaleString("en")} held-out Reddit posts, the model reaches a macro-F1 of{" "}
            <strong>{card.test_macro_f1[0].toFixed(3)}</strong> (95 % CI {card.test_macro_f1[1].toFixed(3)} to{" "}
            {card.test_macro_f1[2].toFixed(3)}) and finds {Math.round(card.test_recall_distress * 100)} % of distress posts.
            Short sentences written to test it tell a different story:
          </p>
          <ChallengeStrip groups={card.challenge_groups} />
          <h2>Limitations</h2>
          <ul className="limits">
            {card.limitations.map((l) => <li key={l}>{l}</li>)}
          </ul>
        </section>
      )}

      <LiveLab />

      <footer className="footer">
        <a href={REPO_URL} target="_blank" rel="noreferrer">Code, data audit and full results on GitHub</a>
        <span>Abd Elmouhib Souri, 2026</span>
      </footer>
    </div>
  );
}
