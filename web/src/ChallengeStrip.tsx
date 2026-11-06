// Challenge-set results (frozen before evaluation): one row per group, correct answers out of n.
import type { ChallengeGroup } from "./api";

// Plain-language names for the group keys returned by the API
const NAMES: Record<string, string> = {
  "a_implicit_distress (label 1)": "Distress without keywords",
  "b_keyword_no_distress (label 0)": "A keyword without distress",
  "c_teen_topic_distress (label 1)": "Distress in a teen context",
  "d_negation (label 0)": "A negation that cancels distress",
  "d_negation (label 1)": "A negation that expresses distress",
};

export function ChallengeStrip({ groups }: { groups: Record<string, ChallengeGroup> }) {
  return (
    <table className="strip">
      <caption className="visually-hidden">Correct answers per challenge group</caption>
      <tbody>
        {Object.entries(groups).map(([key, g]) => {
          const share = g.correct / g.n;
          return (
            <tr key={key} className={share < 0.25 ? "is-weak" : ""}>
              <th scope="row">{NAMES[key] ?? key}</th>
              <td className="strip-bar">
                {/* one cell per sentence: filled = answered correctly */}
                <span className="cells" aria-hidden="true">
                  {Array.from({ length: g.n }, (_, i) => (
                    <span key={i} className={i < g.correct ? "cell on" : "cell"} />
                  ))}
                </span>
              </td>
              <td className="strip-count">{g.correct} of {g.n}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
