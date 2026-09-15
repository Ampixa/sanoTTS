/**
 * StageView — routes the current story chapter to its right-side view.
 * Interaction contract: everything here is playable locally (hover, drag,
 * play, WASM, 3D overview) and NEVER moves the story; only the left
 * column's scroll does.
 */
import { useViz } from "../data/store";
import { HeroView } from "./HeroView";
import { FrontendFlow } from "./FrontendFlow";
import { DurationDetail } from "./DurationDetail";
import { RegulatorFan } from "./RegulatorFan";
import { MelDetail } from "./MelDetail";
import { ConvNextBlock } from "./ConvNextBlock";
import { SpectrumDetail } from "./SpectrumDetail";
import { OlaDiagram, WaveformDetail } from "./OlaWaveform";
import { Overview3D } from "./Overview3D";
import { BudgetChart, DeployChart, EvidenceChart, LanesChart, LiveCard } from "./EvidenceViews";

function WaveformStage() {
  return (
    <>
      <OlaDiagram />
      <WaveformDetail />
    </>
  );
}

const VIEWS: Record<number, () => JSX.Element> = {
  0: HeroView,
  1: FrontendFlow,
  2: DurationDetail,
  3: RegulatorFan,
  4: MelDetail,
  5: ConvNextBlock,
  6: SpectrumDetail,
  7: WaveformStage,
  8: BudgetChart,
  9: LanesChart,
  10: EvidenceChart,
  11: DeployChart,
  12: LiveCard,
};

export function StageView() {
  const chapter = useViz((s) => s.chapter);
  const overviewOpen = useViz((s) => s.overviewOpen);
  const setOverviewOpen = useViz((s) => s.setOverviewOpen);
  const View = VIEWS[chapter] ?? HeroView;
  return (
    <div id="stage-view">
      <div className="stage-fade" key={chapter}>
        <View />
      </div>
      {overviewOpen && chapter !== 0 && (
        <div className="overview-overlay stage-fade">
          <Overview3D focusChapter={chapter} onClose={() => setOverviewOpen(false)} />
        </div>
      )}
    </div>
  );
}
