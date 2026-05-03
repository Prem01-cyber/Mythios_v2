import traceback
import requests
import json
import os
import subprocess
import requests
import zipfile
import gzip
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import hashlib
import time
from dotenv import load_dotenv

load_dotenv()

NVD_API_KEY = os.getenv("NVD_API_KEY")


class vulnerabilityDataCollector:
    def __init__(
        self, base_dir: str = "/home/premjampuram/Projects/Mythos_v2/vuln_data"
    ):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.datasets_dir = self.base_dir / "datasets"

        for dir_path in [self.raw_dir, self.processed_dir, self.datasets_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "VulnDataCollector/1.0 (Educational Research)",
                "apiKey": NVD_API_KEY,
            }
        )

    def download_nvd_feeds(self) -> bool:
        nvd_dir = self.raw_dir / "nvd"
        nvd_dir.mkdir(exist_ok=True)

        base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"

        output_file = nvd_dir / "nvd_cves_all.json"

        print(f"Writing to: {output_file}")

        total_cves = 0
        start_index = 0
        max_results = 500000

        try:
            with open(output_file, "w") as f:
                f.write('{\n  "vulnerabilities": [\n')

                first_batch = True

                while total_cves < max_results:
                    params = {"resultsPerPage": 2000, "startIndex": start_index}

                    try:
                        response = self.session.get(
                            base_url,
                            headers=self.session.headers,
                            params=params,
                            timeout=60,
                        )

                        if response.status_code == 200:
                            data = response.json()

                            if start_index == 0:
                                print(
                                    f"Total results available: {data.get('totalResults', 'unknown')}"
                                )

                            vulnerabilities = data.get("vulnerabilities", [])

                            if not vulnerabilities:
                                print("No more vulnerabilities found.")
                                break

                            for vuln in vulnerabilities:
                                if not first_batch:
                                    f.write(",\n")
                                else:
                                    first_batch = False

                                json.dump(vuln, f, indent=4)
                                total_cves += 1

                            f.flush()
                            print(
                                f"Streamed {len(vulnerabilities)} CVEs (Total: {total_cves})"
                            )

                            total_results = data.get("totalResults", 0)
                            if total_cves >= total_results or total_cves >= max_results:
                                print(f"Reached limit or end of results")
                                break

                            start_index += 2000
                            time.sleep(0.7)
                        else:
                            print(
                                f"Error: {response.status_code} - {response.text[:200]}"
                            )
                            break

                    except Exception as e:
                        print(f"Exception during fetch: {e}")
                        import traceback

                        traceback.print_exc()
                        break

                f.write("\n  ]\n}\n")

            print(f"\nSuccessfully streamed {total_cves} CVEs to {output_file}")
            return total_cves > 0

        except Exception as e:
            print(f"Error opening/writing file: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _extract_cve_info(self, vuln_item: Dict) -> Optional[Dict]:
        try:
            cve = vuln_item.get("cve", {})
            cve_id = cve.get("id", "")
            descriptions = cve.get("descriptions", [])
            description = ""
            for desc in descriptions:
                if desc.get("lang", "en") == "en":
                    description = desc.get("value", "")
                    break

            published_date = cve.get("published", "")
            metrics = cve.get("metrics", {})
            base_score = 0.0
            severity = "UNKNOWN"
            attack_vector = "UNKNOWN"
            attack_complexity = "UNKNOWN"
            exploitability = 0.0

            if "cvssMetricV31" in metrics and metrics["cvssMetricV31"]:
                cvss_v31 = metrics["cvssMetricV31"][0]["cvssData"]
                base_score = cvss_v31.get("baseScore", 0.0)
                severity = cvss_v31.get("baseSeverity", "UNKNOWN")
                attack_vector = cvss_v31.get("attackVector", "UNKNOWN")
                attack_complexity = cvss_v31.get("attackComplexity", "UNKNOWN")
                exploitability = (
                    metrics["cvssMetricV31"][0].get("exploitabilityScore", 0.0) / 10.0
                )
            elif "cvssMetricV30" in metrics and metrics["cvssMetricV30"]:
                cvss_v30 = metrics["cvssMetricV30"][0]["cvssData"]
                base_score = cvss_v30.get("baseScore", 0.0)
                severity = cvss_v30.get("baseSeverity", "UNKNOWN")
                attack_vector = cvss_v30.get("attackVector", "UNKNOWN")
                attack_complexity = cvss_v30.get("attackComplexity", "UNKNOWN")
                exploitability = (
                    metrics["cvssMetricV30"][0].get("exploitabilityScore", 0.0) / 10.0
                )
            elif "cvssMetricV2" in metrics and metrics["cvssMetricV2"]:
                cvss_v2 = metrics["cvssMetricV2"][0]["cvssData"]
                base_score = cvss_v2.get("baseScore", 0.0)
                severity = cvss_v2.get("baseSeverity", "UNKNOWN")
                attack_vector = cvss_v2.get("attackVector", "UNKNOWN")
                attack_complexity = cvss_v2.get("attackComplexity", "UNKNOWN")
                exploitability = (
                    metrics["cvssMetricV2"][0].get("exploitabilityScore", 0.0) / 10.0
                )
            weaknesses = cve.get("weaknesses", [])
            cwes = []
            for weakness in weaknesses:
                for desc in weakness.get("description", []):
                    cwe_value = desc.get("value", "")
                    if cwe_value.startswith("CWE-"):
                        cwes.append(cwe_value)

            cwe_id = cwes[0] if cwes else "CWE-Unknown"
            configurations = cve.get("configurations", {})
            affected_products = self._extract_affected_products_v2(configurations)
            is_windows = any(
                "windows" in prod.lower() or "microsoft" in prod.lower()
                for prod in affected_products
            )

            return {
                "cve_id": cve_id,
                "description": description,
                "published_date": published_date,
                "cvss_score": base_score,
                "severity": severity,
                "attack_vector": attack_vector,
                "attack_complexity": attack_complexity,
                "cwe_id": cwe_id,
                "cwe_list": ",".join(cwes),
                "affected_products": ";".join(affected_products[:5]),
                "is_windows": is_windows,
                "exploitability": exploitability,
            }

        except Exception as e:
            print(f"Warning: Error extracting CVE info: {e}")
            return None

    def _extract_affected_products_v2(self, configurations: List) -> List[str]:
        products = []

        try:
            for config in configurations:
                nodes = config.get("nodes", [])
                for node in nodes:
                    cpe_matches = node.get("cpeMatch", [])
                    for match in cpe_matches:
                        if match.get("vulnerable", False):
                            cpe = match.get("criteria", "")
                            parts = cpe.split(":")
                            if len(parts) >= 5:
                                vendor = parts[3]
                                product = parts[4]
                                products.append(f"{vendor}/{product}")
        except:
            pass

        return list(set(products))

    def process_nvd_feeds(self) -> pd.DataFrame:
        print("\n" + "=" * 60)
        print("Processing NVD CVE Data")
        print("=" * 60)

        nvd_dir = self.raw_dir / "nvd"
        json_file = nvd_dir / "nvd_cves_all.json"

        print(f"\nReading from: {json_file}")

        if not json_file.exists():
            print("Error: Input file not found")
            return pd.DataFrame()

        all_cves = []

        try:
            print("Loading JSON data...")
            with open(json_file, "r") as f:
                data = json.load(f)

            vulnerabilities = data.get("vulnerabilities", [])
            total_vulns = len(vulnerabilities)
            print(f"Found {total_vulns} vulnerabilities to process")

            processed_count = 0
            skipped_count = 0

            for i, vuln_item in enumerate(vulnerabilities, 1):
                if i % 10000 == 0:
                    print(f"Progress: {i}/{total_vulns} ({(i/total_vulns*100):.1f}%)")

                cve_data = self._extract_cve_info(vuln_item)
                if cve_data:
                    all_cves.append(cve_data)
                    processed_count += 1
                else:
                    skipped_count += 1

            print(f"\nProcessing complete:")
            print(f"  Processed: {processed_count}")
            print(f"  Skipped: {skipped_count}")

        except Exception as e:
            print(f"Error during processing: {e}")
            traceback.print_exc()

        df = pd.DataFrame(all_cves)
        output_path = self.processed_dir / "nvd_cves.csv"

        print(f"\nWriting to: {output_path}")
        df.to_csv(output_path, index=False)
        print(f"Successfully saved {len(df)} CVEs to CSV")

        return df

    def create_training_splits(self):
        input_path = self.processed_dir / "nvd_cves.csv"
        df = pd.read_csv(input_path)
        df_filtered = df[
            (df["cvss_score"] > 0)
            & (df["description"].str.len() > 50)
            & (df["cwe_id"] != "CWE-Unknown")
        ].copy()
        df_filtered["published_date"] = pd.to_datetime(df_filtered["published_date"])
        df_filtered = df_filtered.sort_values("published_date")
        n = len(df_filtered)
        train_end = int(0.70 * n)
        val_end = int(0.85 * n)
        train_df = df_filtered.iloc[:train_end]
        val_df = df_filtered.iloc[train_end:val_end]
        test_df = df_filtered.iloc[val_end:]
        train_df.to_csv(self.processed_dir / "train.csv", index=False)
        val_df.to_csv(self.processed_dir / "val.csv", index=False)
        test_df.to_csv(self.processed_dir / "test.csv", index=False)
        print(f"Successfully created training, validation, and test splits")

        stats = {
            "total_samples": len(train_df),
            "severity_distribution": train_df["severity"].value_counts().to_dict(),
            "cwe_distribution": train_df["cwe_id"].value_counts().head(20).to_dict(),
            "cvss_mean": float(train_df["cvss_score"].mean()),
            "cvss_std": float(train_df["cvss_score"].std()),
            "windows_pct": float(train_df["is_windows"].sum() / len(train_df) * 100),
        }

        with open(self.datasets_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=4)
        print(f"Successfully saved stats to {self.datasets_dir / 'stats.json'}")
        return train_df, val_df, test_df


if __name__ == "__main__":
    collector = vulnerabilityDataCollector()
    collector.download_nvd_feeds()
    collector.process_nvd_feeds()
    collector.create_training_splits()
