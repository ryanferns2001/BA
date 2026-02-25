from blackboard.codebase.core import blackboard_semantic_mapping
from simplellm.codebase.core import paper_agents as simplellm
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VC_SLAM_BASE_PATH = ROOT / "datacorpus" / "vcslam"
VC_SLAM_BASE = str(VC_SLAM_BASE_PATH)

full_sample_ids = [
            "0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009", "0010",
            "0011", "0012", "0013", "0014", "0015", "0016", "0017", "0018", "0019", "0020",
            "0021", "0022", "0023", "0024", "0025", "0026", "0027", "0028", "0029", "0030",
            "0031", "0032", "0033", "0034", "0035", "0036", "0037", "0038", "0039", "0040",
            "0041", "0042", "0043", "0044", "0045", "0046", "0047", "0048", "0049", "0050",
            "0051", "0052", "0053", "0054", "0055", "0056", "0057", "0058", "0059", "0060",
            "0061", "0062", "0063", "0064", "0065", "0066", "0067", "0068", "0069", "0070",
            "0071", "0072", "0073", "0074", "0075", "0076", "0077", "0078", "0079", "0080",
            "0081", "0082", "0083", "0084", "0085", "0086", "0087", "0088", "0089", "0090",
            "0091", "0092", "0093", "0094", "0095", "0096", "0097", "0098", "0099", "0100",
            "0101"
        ]

sample_ids = full_sample_ids
historical_ids = ["0001", "0010", "0020", "0030", "0040", "0050", "0060", "0070", "0080", "0090", "0100"]

def main():
    export_path_simple_llm_str = str(ROOT / "simplellm" / "exports")
    export_path_blackboard_str = str(ROOT / "blackboard" / "exports")

    #simplellm.main(vcslam_path=VC_SLAM_BASE, sample_ids=sample_ids, historical_ids=historical_ids, base_export_dir=export_path_simple_llm_str,evaluation_run=True)
    blackboard_semantic_mapping.main(vcslam_path=VC_SLAM_BASE, sample_ids=sample_ids, historical_ids=historical_ids, export_path=export_path_blackboard_str, evaluation_run=True)

if __name__ == '__main__':
    main()
