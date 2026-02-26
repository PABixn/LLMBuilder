import type { Dispatch, SetStateAction } from "react";

import type { StudioDocument, StudioDocumentNumericField } from "../../types";
import { integerInputValue, parseIntegerInput } from "../../utils/format";

type BaseModelPanelProps = {
  documentState: StudioDocument;
  updateBaseField: (key: StudioDocumentNumericField, value: number) => void;
  setDocumentState: Dispatch<SetStateAction<StudioDocument>>;
};

export function BaseModelPanel({
  documentState,
  updateBaseField,
  setDocumentState,
}: BaseModelPanelProps) {
  return (
    <section id="base-model" className="panelCard">
      <div className="panelHead">
        <div>
          <p className="panelEyebrow">Base Model</p>
          <h2>Base dimensions</h2>
          <p className="panelCopy">
            Shared dimensions used by the builder and validation checks.
          </p>
        </div>
      </div>
      <div className="fieldGrid">
        <label className="fieldLabel" htmlFor="context_length">
          <span>Context Length</span>
          <input
            id="context_length"
            type="number"
            min={1}
            step={1}
            value={integerInputValue(documentState.context_length)}
            onChange={(event) =>
              updateBaseField(
                "context_length",
                parseIntegerInput(event.target.value, documentState.context_length)
              )
            }
          />
        </label>
        <label className="fieldLabel" htmlFor="vocab_size">
          <span>Vocab Size</span>
          <input
            id="vocab_size"
            type="number"
            min={1}
            step={1}
            value={integerInputValue(documentState.vocab_size)}
            onChange={(event) =>
              updateBaseField(
                "vocab_size",
                parseIntegerInput(event.target.value, documentState.vocab_size)
              )
            }
          />
        </label>
        <label className="fieldLabel" htmlFor="n_embd">
          <span>Embedding Dimension</span>
          <input
            id="n_embd"
            type="number"
            min={1}
            step={1}
            value={integerInputValue(documentState.n_embd)}
            onChange={(event) =>
              updateBaseField("n_embd", parseIntegerInput(event.target.value, documentState.n_embd))
            }
          />
        </label>
        <label className="toggleField tall" htmlFor="weight_tying">
          <input
            id="weight_tying"
            type="checkbox"
            checked={documentState.weight_tying}
            onChange={(event) =>
              setDocumentState((current) => ({ ...current, weight_tying: event.target.checked }))
            }
          />
          <span>Weight Tying</span>
        </label>
      </div>
    </section>
  );
}
