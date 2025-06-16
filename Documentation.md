# CynsesAI PCAP Analyzer Documentation

## 1. Project Overview and Goals

CynsesAI PCAP Analyzer is a comprehensive network security analysis tool designed to process PCAP (Packet Capture) files, identify potential threats, and provide detailed insights into network traffic. It automates the analysis of network data by leveraging powerful open-source tools, enriching findings with threat intelligence, detecting anomalies, and generating interactive visualizations and comprehensive reports.

The primary goals of the project are:
- To provide a robust and automated solution for PCAP analysis.
- To detect and identify various network threats, including malware, intrusions, and anomalous behavior.
- To enrich network event data with external threat intelligence for better context and accuracy.
- To offer clear, actionable insights through comprehensive reports and visualizations.
- To create a flexible and extensible platform for network security analysis that can be customized by users.

## 2. In-depth Features Description

The CynsesAI PCAP Analyzer offers a rich set of features:

*   **PCAP Analysis with Suricata and Zeek:**
    *   **Suricata:** Utilizes the Suricata Intrusion Detection System (IDS) to inspect network traffic against known threat signatures and rules. It identifies and logs alerts for suspicious activities, malware communications, and policy violations.
    *   **Zeek:** Employs the Zeek Network Security Monitor to perform deep packet inspection and generate detailed logs of network connections, application-layer protocols (HTTP, DNS, FTP, SMTP, SSL, etc.), and file transfers.
*   **Threat Intelligence Enrichment:**
    *   Correlates IP addresses found in network logs (primarily from Suricata alerts) with external threat intelligence feeds.
    *   Currently supports AbuseIPDB and VirusTotal to check the reputation and history of IPs, identifying connections to known malicious actors or infrastructure.
*   **Anomaly Detection:**
    *   Analyzes Suricata alert patterns to identify unusual or unexpected behaviors that might indicate sophisticated attacks or zero-day threats.
    *   Leverages Large Language Models (LLMs) to assess chunks of Suricata alerts for potential security threats and their severity.
*   **Network Traffic Classification:**
    *   Employs a pre-trained BERT-based machine learning model (`rdpahalavan/bert-network-packet-flow-header-payload`) to classify network traffic into various categories, including normal traffic and different types of attacks (e.g., DoS, Backdoor, Exploits, Reconnaissance).
    *   This feature is optional and can be enabled by the user at runtime.
*   **Interactive Attack Visualization:**
    *   Generates various graphs and visualizations to help understand network interactions and potential attack paths.
    *   **NetworkX Static Graph:** Creates a static image (`enterprise_attack_graph.png`) depicting nodes (IP addresses) and edges (connections/alerts), with nodes colored and sized based on threat scores.
    *   **Plotly Interactive Graph:** Produces an HTML-based interactive graph (`interactive_attack_graph.html`) allowing users to zoom, pan, and hover over nodes and edges for more details.
    *   **Security Summary HTML:** Generates an HTML report (`security_summary.html`) with charts summarizing alert severities, categories, connection sizes, and threat score distributions.
*   **Workflow Orchestration with LangGraph:**
    *   Uses LangGraph, a library for building stateful, multi-actor applications with LLMs, to manage the complex multi-step analysis process. The workflow is defined as a graph of interconnected nodes, each performing a specific task.
*   **AI-Powered Analysis (LLMs):**
    *   Leverages Large Language Models (LLMs) like OpenAI's models (via Chutes AI proxy) or local Ollama models (e.g., llama3.2) for:
        *   Enhanced interpretation of Suricata alerts and Zeek logs.
        *   Analysis of threat intelligence data.
        *   Summarization of anomalies.
        *   Generation of the final comprehensive report.
    *   Includes a RAG (Retrieval Augmented Generation) component using Suricata documentation to provide context to the LLM for more accurate analysis of Suricata events.
*   **Comprehensive Reporting:**
    *   Produces a detailed analysis report in Markdown format (`pcap_analysis_report.md`).
    *   The report includes an executive summary, technical findings, threat assessment, recommended actions, and potentially Mitre ATT&CK mapping.
*   **Mermaid Diagram Generation:**
    *   Creates attack flow diagrams in Mermaid syntax, rendered as a PNG image (`attack_flow.png`). These diagrams visually represent detected attack sequences and relationships between internal and external entities.
*   **Caching Mechanism:**
    *   Utilizes `diskcache` to cache results from LLM calls and threat intelligence lookups. This speeds up subsequent analyses of the same data or IP addresses, reducing redundant API calls and processing time.

## 3. System Architecture (Conceptual)

The CynsesAI PCAP Analyzer is designed as a modular system orchestrated by a central workflow manager.

```
+-----------------------+      +-----------------------+      +-----------------------+
|     Input PCAP File   |----->|   Data Ingestion &    |----->|  Core Analysis Engines|
+-----------------------+      |   Initialization      |      +-----------------------+
                               |(main.py, cleanup)     |      | - Suricata            |
                               +-----------------------+      | - Zeek                |
                                                              +-----------------------+
                                                                        |
                                                                        V
+---------------------------------+      +---------------------------------+      +-----------------------------+
| Enrichment & Contextualization  |<-----|      Data Aggregation &         |<-----|   Post-Processing &         |
+---------------------------------+      |      Normalization              |      |   Further Analysis          |
| - Threat Intelligence (APIs)    |      +---------------------------------+      +-----------------------------+
| - RAG (Suricata Docs)           |      | (Consolidation of Suricata/Zeek |      | - Network Traffic Classifier|
+---------------------------------+      |  outputs, IP extraction)        |      | - Protocol Analysis (LLM)   |
          |                              +---------------------------------+      | - Anomaly Detection (LLM)   |
          |                                              |                        +-----------------------------+
          V                                              V                                      |
+----------------------------------+     +----------------------------------+                    |
| AI-Powered Analysis & Decision   |<----|      Visualization Generation      |<-------------------+
+----------------------------------+     +----------------------------------+
| - LLM for insights, summaries    |     | - NetworkX, Plotly, Mermaid      |
| - LangGraph for workflow mgmt    |     +----------------------------------+
+----------------------------------+
          |
          V
+----------------------------------+
|      Output Generation           |
+----------------------------------+
| - Markdown Report                |
| - HTML Visualizations            |
| - Logs (Suricata, Zeek)          |
+----------------------------------+
```

**Key Components:**

1.  **Input:** A PCAP file containing network traffic.
2.  **Orchestrator (`main.py` with LangGraph):** Manages the overall workflow, defining the sequence and dependencies of analysis tasks.
3.  **Core Analysis Engines:**
    *   **Suricata Module (`suricata_parser.py`):** Executes Suricata, parses its `eve.json` and `fast.log` outputs.
    *   **Zeek Module (`zeek.py`):** Executes Zeek, collects and parses its various log files (e.g., `conn.log`, `http.log`, `dns.log`).
4.  **Processing Modules (`modules/` directory):**
    *   **Network Traffic Classifier (`network_traffic_classifier.py`):** Classifies traffic using a ML model.
    *   **Protocol Analyzer (`protocol_analysis.py`):** Uses LLMs to analyze protocol data from Zeek logs.
    *   **Threat Intelligence (`threat_intel.py`):** Queries AbuseIPDB and VirusTotal.
    *   **Anomaly Detector (`anomaly_detection.py`):** Uses LLMs to find anomalies in Suricata alerts.
    *   **Visualization (`visualization.py`):** Generates graphs and HTML summaries.
    *   **Report Generator (`report_generation.py`):** Compiles the final Markdown report using LLM.
5.  **Configuration (`config/settings.py`):** Stores API keys, paths, and other operational parameters.
6.  **LLM Integration:** Interacts with LLMs (OpenAI via Chutes or Ollama) for advanced analysis and text generation.
7.  **Caching (`diskcache`):** Stores results of expensive operations like API calls and LLM inferences.
8.  **Outputs:** Generates reports, visualizations, and raw tool logs.

## 4. Detailed Workflow Explanation

The analysis workflow is managed by `main.py` using LangGraph, defining a stateful graph where each node represents an analysis step. The state (`AnalysisState`) is passed between nodes, accumulating data.

**Workflow Graph (conceptual, refer to `workflow_graph.png` for actual image):**

The `README.md` describes the workflow as:

1.  **Initialization (`initialize_analysis` node):**
    *   Cleans up resources from previous runs (e.g., Suricata/Zeek output directories, old graphs, cache).
    *   Sets up the analysis environment.
    *   Retrieves relevant Suricata documentation using RAG to provide context for later LLM analysis.
    *   Input: `pcap_path`
    *   Output: Initial `AnalysisState` with `pcap_path` and `rag_context`.

2.  **Parallel Analysis - Suricata & Zeek:**
    *   **Suricata Analysis (`run_suricata_node`):**
        *   Runs Suricata on the PCAP file using `modules.suricata_parser.run_suricata()`.
        *   Parses `eve.json` to extract Suricata events.
        *   Updates `rag_context` based on detected event types.
        *   Output: `suricata_events`, updated `rag_context`.
    *   **Zeek Analysis (`run_zeek_node`):**
        *   Runs Zeek on the PCAP file using `modules.zeek.run_zeek()`.
        *   Collects various Zeek logs (conn.log, dns.log, etc.).
        *   Output: `zeek_logs`.

3.  **Merge Results (`merge_results` node):**
    *   A conceptual merge point in LangGraph. Data from Suricata and Zeek nodes are now available in the shared `AnalysisState`.

4.  **Network Traffic Classification (Conditional Node - `network_traffic_classification_node`):**
    *   The user is prompted (`ask_run_classifier` function in `main.py`) whether to run this step.
    *   If yes:
        *   Uses `modules.network_traffic_classifier.predictingRowsCategoryOnGPU()` to classify packets from the PCAP file.
        *   The results (counts of non-normal traffic types) are stored in `packets_brief` dictionary within the classifier module and then added to the `AnalysisState`.
    *   Output: `traffic_classification` (if run).

5.  **Protocol Analysis (`protocol_analysis_node`):**
    *   Takes Zeek logs from `AnalysisState`.
    *   Uses `modules.protocol_analysis.analyze_protocols()` which leverages an LLM to:
        *   Analyze protocol distribution and usage.
        *   Identify unusual ports or patterns.
        *   Flag suspicious HTTP requests or DNS queries.
        *   Correlate findings with Suricata alerts (using `rag_context`).
    *   Output: `protocol_analysis` (containing raw samples and LLM analysis).

6.  **Threat Intelligence Enrichment (`threat_intel_node`):**
    *   Extracts unique IP addresses from Suricata events.
    *   For each IP, queries AbuseIPDB and VirusTotal using `modules.threat_intel.real_threat_intel()`.
    *   Uses an LLM to analyze the collected threat data in conjunction with `rag_context`.
    *   Output: `threat_intel` (containing raw API responses and LLM analysis).

7.  **Anomaly Detection (`anomaly_detection_node`):**
    *   Parses Suricata `fast.log` using `modules.anomaly_detection.parse_fastlog()`.
    *   Processes alerts in chunks using `modules.anomaly_detection.process_alerts()` which invokes an LLM to:
        *   Describe potential threats and their severity based on alert clusters.
    *   Output: `anomalies` (list of LLM-generated anomaly descriptions), `anomaly_summary`.

8.  **Visualization (`visualization_node`):**
    *   Parses Suricata `fast.log` and Zeek `conn.log` again (note: could potentially use already parsed data from state).
    *   Extracts packet data using Scapy (`extract_packet_data_with_scapy`).
    *   Builds an enhanced attack graph using NetworkX (`build_enhanced_attack_graph`), incorporating threat scores for nodes.
    *   Generates:
        *   Static NetworkX plot (`attack_graph.png`).
        *   Interactive Plotly graph (`interactive_attack_graph.html`).
    *   Output: `visualization_data` (containing the graph object and paths to image files).

9.  **Report Generation (`report_generation_node`):**
    *   Generates a Mermaid diagram code for attack flow based on Suricata events and Zeek connection logs (`generate_attack_diagram` and `render_diagram` in `main.py`). The image is saved as `attack_flow.png`.
    *   Summarizes all collected data: Suricata events, protocol analysis, threat intelligence, anomalies, visualization plan, traffic classification, and RAG context.
    *   Constructs a detailed prompt for an LLM to generate the final Markdown report (`pcap_analysis_report.md`).
    *   The report includes sections like Executive Summary, Technical Findings (with Mermaid diagram), Attack Flow Analysis, Threat Assessment, Recommended Actions, Mitre ATT&CK Mapping, and Threat Classification.
    *   Output: `report` (the Markdown content), `diagram_code`, `diagram_image`.

10. **End:** The workflow concludes. The final report is saved to `pcap_analysis_report.md`. An interactive GPT session (`interactive_gpt_session` in `main.py`) is then initiated to allow the user to ask questions about the analysis results.

The state transitions are defined in `main.py` by adding edges between these nodes in the `workflow` StateGraph object.

## 5. Key Technologies Utilized

*   **Packet Analysis Engines:**
    *   **Suricata:** High-performance Network IDS, IPS, and Network Security Monitoring engine. Used for rule-based threat detection.
    *   **Zeek (formerly Bro):** Powerful framework for network analysis and security monitoring. Used for generating detailed logs of network activity.
    *   **Scapy:** Python library for packet manipulation, used in `visualization.py` to extract data directly from PCAP for graph building.
*   **Data Handling & Manipulation:**
    *   **Python:** The core programming language for the project.
    *   **Pandas:** Used in `visualization.py` for robust parsing of Zeek's `conn.log`.
*   **Visualization:**
    *   **NetworkX:** Python library for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks. Used for creating the base attack graphs.
    *   **Matplotlib:** Python plotting library, used by NetworkX for rendering static graph images.
    *   **Plotly:** Interactive graphing library. Used to create dynamic HTML-based network visualizations and summary charts.
    *   **Mermaid (via `mmdc` - Mermaid CLI):** JavaScript-based diagramming and charting tool. Used to generate attack flow diagrams from a textual description.
*   **Workflow & AI:**
    *   **LangGraph:** Library for building stateful, multi-actor applications with LLMs, particularly useful for orchestrating chains and agents. Used in `main.py` to define and manage the analysis workflow.
    *   **Langchain:** Framework for developing applications powered by language models. Used for RAG, LLM interactions, and prompt management.
    *   **Large Language Models (LLMs):**
        *   **OpenAI Models (GPT series):** Accessed via Chutes AI API (`CHUTES_API_BASE`, `CHUTES_API_KEY`) for tasks like report generation, protocol analysis, and anomaly detection. Specifically, `deepseek-ai/DeepSeek-V3-0324` and `deepseek-ai/deepseek-chat` are mentioned.
        *   **Ollama Models (e.g., `llama3.2`):** Can be used as an alternative for local LLM processing if `USE_OLLAMA` is set to `True` in `main.py`.
    *   **Transformers (Hugging Face):** Used by `network_traffic_classifier.py` to load and run the pre-trained BERT model for traffic classification.
    *   **PyTorch:** The backend framework for the Hugging Face Transformers model used in traffic classification.
*   **Threat Intelligence APIs:**
    *   **AbuseIPDB:** API for checking IP address reputation.
    *   **VirusTotal:** API for inspecting items for malware and checking IP/domain reputation.
*   **Caching:**
    *   **DiskCache:** Disk-backed cache library for Python. Used to cache results of expensive function calls, notably threat intelligence queries and potentially LLM responses (though LLM caching seems commented out in some places).
*   **Configuration:**
    *   Python scripts (`config/settings.py`) are used for managing settings like API keys, file paths, and model choices.

## 6. Module-by-Module Breakdown

The project is structured into several Python modules, primarily located in the `modules/` directory, with `main.py` as the orchestrator and `config/settings.py` for configuration.

*   **`main.py`:**
    *   **Purpose:** The central script that defines and executes the entire PCAP analysis workflow.
    *   **Key Functions:**
        *   Sets up LangGraph state (`AnalysisState`) and workflow graph.
        *   Defines nodes for each step of the analysis (initialization, Suricata, Zeek, classification, protocol analysis, threat intel, anomaly detection, visualization, report generation).
        *   Manages the flow of data between nodes.
        *   Handles user interaction (e.g., asking to run the classifier).
        *   Initializes LLMs (OpenAI via Chutes or Ollama).
        *   Sets up RAG with Suricata documentation.
        *   Includes functions for generating and rendering Mermaid attack flow diagrams.
        *   Saves the final report and initiates an interactive GPT session.
    *   **Dependencies:** All modules in `modules/`, LangGraph, Langchain, LLM libraries, DiskCache, `config/settings.py`.

*   **`config/settings.py`:**
    *   **Purpose:** Stores all configuration variables for the application.
    *   **Contents:**
        *   `PCAP_FILE`: Path to the input PCAP file.
        *   `USE_OLLAMA`: Boolean to switch between OpenAI/Chutes and local Ollama LLMs.
        *   `CHUTES_API_KEY`, `CHUTES_API_BASE`: Credentials for Chutes AI (acting as OpenAI proxy).
        *   `SURICATA_DOCS_URL`: URL for Suricata documentation used by RAG.
        *   `SURICATA_FAST_LOG`, `ZEEK_CONN_LOG`: Paths to specific log files (may need updates if outputs are consistently in `SURICATA_OUTPUT_DIR` and `ZEEK_OUTPUT_DIR`).
        *   `SURICATA_CONFIG`: Path to Suricata configuration file (`suricata.yaml`).
        *   `SURICATA_OUTPUT_DIR`, `ZEEK_OUTPUT_DIR`: Directories for Suricata and Zeek outputs.
        *   `ABUSEIPDB_API_KEY`, `VIRUSTOTAL_API_KEY`: API keys for threat intelligence services.
        *   `GRAPH_OUTPUT_DIR`: Directory for generated graph images.
    *   **Note:** The `README.md` mentions `SURICATA_RULES_DIR` which is not present in the provided `settings.py` but is a typical Suricata configuration parameter.

*   **`modules/suricata_parser.py`:**
    *   **Purpose:** To run Suricata on the PCAP file and parse its primary output.
    *   **Key Functions:**
        *   `run_suricata(pcap_path)`: Executes the `suricata` command-line tool with the specified PCAP, configuration, and output directory. Parses the `eve.json` file line by line, converting JSON objects into a list of Python dictionaries (events).
    *   **Outputs:** List of Suricata event dictionaries. Raw Suricata logs are stored in `suricata_output/`.
    *   **Dependencies:** `subprocess`, `json`, `os`, `config.settings` (for `SURICATA_CONFIG`). DiskCache is imported but its usage is commented out.

*   **`modules/zeek.py`:**
    *   **Purpose:** To run Zeek on the PCAP file and collect its log files.
    *   **Key Functions:**
        *   `run_zeek(pcap_path)`: Executes the `zeek` command-line tool with the specified PCAP and directs logs to the `zeek_output/` directory. Reads all `.log` files from this directory into a dictionary where keys are filenames and values are lists of lines from the file.
    *   **Outputs:** Dictionary of Zeek logs. Raw Zeek logs are stored in `zeek_output/`.
    *   **Dependencies:** `subprocess`, `os`, `pathlib`. DiskCache is imported but its usage is commented out.

*   **`modules/network_traffic_classifier.py`:**
    *   **Purpose:** Classifies network traffic from the PCAP file using a pre-trained BERT model.
    *   **Key Functions:**
        *   `processing_packet_conversion(packet)`: Converts Scapy packet objects into a string of decimal features suitable for the BERT model.
        *   `predictingRowsCategoryOnGPU(file_path, ...)`: Reads packets from the PCAP, converts them, and uses the loaded Hugging Face model to predict traffic categories. Stores counts of non-normal traffic in the global `packets_brief` dictionary. A CPU version (`predictingRowsCategory`) also exists.
    *   **Model:** `rdpahalavan/bert-network-packet-flow-header-payload` from Hugging Face.
    *   **Outputs:** Populates the `packets_brief` dictionary (e.g., `{'DDoS': 10, 'Port Scan': 5}`).
    *   **Dependencies:** `transformers`, `torch`, `scapy`.

*   **`modules/protocol_analysis.py`:**
    *   **Purpose:** Performs high-level analysis of network protocols using Zeek logs and an LLM.
    *   **Key Functions:**
        *   `analyze_protocols(logs, rag_context)`: Takes Zeek logs and RAG context. Samples lines from each log file, creates a prompt for an LLM to identify suspicious patterns, and returns the raw samples and the LLM's analysis.
    *   **Outputs:** Dictionary containing raw log samples and LLM-generated analysis text.
    *   **Dependencies:** `langchain_openai` (ChatOpenAI), `config.settings` (for LLM API keys).

*   **`modules/threat_intel.py`:**
    *   **Purpose:** Enriches IP addresses with threat intelligence from external APIs.
    *   **Key Functions:**
        *   `real_threat_intel(ip)`:
            *   `query_abuseipdb(ip)`: Queries AbuseIPDB API for the given IP.
            *   `query_virustotal(ip)`: Queries VirusTotal API for the given IP.
            *   Returns a dictionary containing results from both services.
    *   **Outputs:** Dictionary with 'virustotal' and 'abuseipdb' keys, containing API responses.
    *   **Dependencies:** `requests`, `config.settings` (for API keys). DiskCache is imported but its usage is commented out.

*   **`modules/anomaly_detection.py`:**
    *   **Purpose:** Analyzes Suricata `fast.log` alerts to detect anomalies using an LLM.
    *   **Key Functions:**
        *   `parse_fastlog(suricata_output_dir)`: Parses the `fast.log` file (text-based alert log) using regex into a list of structured alert dictionaries.
        *   `chunk_alerts(alerts, chunk_size)`: Splits the list of alerts into smaller chunks.
        *   `analyze_chunk(chunk, llm)`: Sends a chunk of alerts to an LLM and asks it to describe potential threats and severity.
        *   `process_alerts(alerts)`: Asynchronously processes all alert chunks using `analyze_chunk`.
    *   **Outputs:** A dictionary containing a list of LLM-generated anomaly descriptions under the key "anomalies".
    *   **Dependencies:** `re`, `asyncio`, `langchain_openai`, `config.settings`.

*   **`modules/visualization.py`:**
    *   **Purpose:** Generates various network graphs and visual summaries.
    *   **Key Functions:**
        *   `parse_suricata_fast_log(path)`: Parses Suricata `fast.log` (text format).
        *   `parse_zeek_conn_log(path)`: Parses Zeek `conn.log` using Pandas.
        *   `extract_packet_data_with_scapy(pcap_path)`: Reads PCAP using Scapy and extracts flow information.
        *   `get_threat_score(ip)`: Queries AbuseIPDB for an IP's threat score (Note: This is a simplified version compared to `threat_intel.py`, only using AbuseIPDB and a numeric score).
        *   `build_enhanced_attack_graph(alerts, connections, packets)`: Creates a NetworkX `DiGraph` object, adding nodes (IPs) with threat scores and edges representing alerts, connections, or packet flows.
        *   `generate_networkx_plot(G, filename)`: Saves the NetworkX graph as a static PNG image using Matplotlib.
        *   `generate_plotly_interactive(G, filename)`: Saves the NetworkX graph as an interactive HTML file using Plotly.
        *   `generate_summary_report(alerts, connections, packets, filename)`: Creates an HTML summary (`security_summary.html`) with Plotly charts for alert severities, categories, etc.
    *   **Outputs:** Image files (`.png`), HTML files (`.html`) in `attack_graphs/` (or `GRAPH_OUTPUT_DIR`).
    *   **Dependencies:** `networkx`, `matplotlib`, `plotly`, `scapy`, `pandas`, `ipaddress`, `requests`, `config.settings`.

*   **`modules/report_generation.py`:**
    *   **Purpose:** Generates the final comprehensive analysis report using an LLM.
    *   **Key Functions:**
        *   `generate_report(data)`: Takes a dictionary of all analysis findings (Suricata events, protocol analysis, threat intel, anomalies), constructs a detailed prompt, and invokes an LLM to produce the final Markdown report.
    *   **Outputs:** A string containing the Markdown report.
    *   **Dependencies:** `langchain_openai`, `config.settings`.

*   **`Rules/` directory:**
    *   Contains Suricata rule files (e.g., `emerging-all.rules`, `suricata.rules`). `Rules.tar.gz` suggests a packaged set of rules. These are used by Suricata for threat detection.

*   **`cache_dir/` and `modules/cache_dir/`:**
    *   Directories used by `diskcache` to store cached data from API calls or other expensive computations.

## 7. Configuration Options

Configuration is primarily managed in `config/settings.py`.

*   **`PCAP_FILE`**: String.
    *   Path to the PCAP file to be analyzed.
    *   Example: `"/Users/macbook/Desktop/CynsesAI/GoldenEye.pcap"`
*   **`USE_OLLAMA`**: Boolean.
    *   Determines which LLM to use. `False` for Chutes AI (OpenAI proxy), `True` for a local Ollama instance.
    *   Default: `False`
*   **`CHUTES_API_KEY`**: String.
    *   API key for the Chutes AI service.
*   **`CHUTES_API_BASE`**: String.
    *   Base URL for the Chutes AI API.
    *   Example: `"https://llm.chutes.ai/v1"`
*   **`SURICATA_DOCS_URL`**: String.
    *   URL for the Suricata documentation, used for building the RAG vector store.
    *   Example: `"https://suricata.readthedocs.io/en/latest/"`

**File Paths (from `config/settings.py` and inferred defaults in modules):**

*   **`SURICATA_CONFIG`**: String (in `config/settings.py`).
    *   Path to the `suricata.yaml` configuration file.
    *   The `README.md` mentions this, and it's used by `modules/suricata_parser.py`.
*   **`SURICATA_OUTPUT_DIR`**: String.
    *   Directory where Suricata saves its output files (e.g., `eve.json`, `fast.log`).
    *   Default in `suricata_parser.py` is `"suricata_output"`.
    *   Mentioned in `README.md` and `config/settings.py` (though not explicitly used from settings in `suricata_parser.py` directly, `main.py` might use it for cleanup).
*   **`SURICATA_FAST_LOG`**: String (in `config/settings.py`).
    *   Path to Suricata's `fast.log` file.
    *   Used by `visualization.py` and `anomaly_detection.py` (via `main.py` which passes `SURICATA_OUTPUT_DIR` to `parse_fastlog`).
*   **`ZEEK_OUTPUT_DIR`**: String.
    *   Directory where Zeek saves its log files.
    *   Default in `zeek.py` is `"zeek_output"`.
    *   Mentioned in `README.md` and `config/settings.py`.
*   **`ZEEK_CONN_LOG`**: String (in `config/settings.py`).
    *   Path to Zeek's `conn.log` file.
    *   Used by `visualization.py`.
*   **`GRAPH_OUTPUT_DIR`**: String (in `config/settings.py`).
    *   Directory where generated graph images and HTML files are stored.
    *   Default in `visualization.py` is `"attack_graphs"`. Used by `main.py` for cleanup.
*   **`SURICATA_RULES_DIR`**: String (mentioned in `README.md` but not `settings.py`).
    *   Path to the directory containing Suricata rules. This is a standard Suricata setting usually configured within `suricata.yaml`.

**Threat Intelligence API Keys (from `config/settings.py`):**

*   **`ABUSEIPDB_API_KEY`**: String.
    *   API key for AbuseIPDB.
*   **`VIRUSTOTAL_API_KEY`**: String.
    *   API key for VirusTotal. (Note: `README.md` mentions it might need explicit implementation in `threat_intel.py` if not fully present, but the code shows it is used).

**LLM Configuration (in `main.py` and `modules/*.py`):**

*   **Model Selection:**
    *   If `USE_OLLAMA` is `True`: `OllamaLLM(model="llama3.2")` and `OllamaEmbeddings(model="llama3.2")` are used.
    *   If `USE_OLLAMA` is `False`: `ChatOpenAI` with `model="deepseek-ai/DeepSeek-V3-0324"` (or `deepseek-ai/deepseek-chat` in `report_generation.py`) is used via Chutes AI.
*   **LLM Parameters:** Temperature, max_tokens are set during LLM initialization in `main.py` and modules like `anomaly_detection.py`, `protocol_analysis.py`, `report_generation.py`.

**Anomaly Detection (in `modules/anomaly_detection.py`):**

*   **`CHUNK_SIZE`**: Integer.
    *   Number of Suricata alerts to process in a single batch by the LLM.
    *   Default: `5`

**Caching:**
*   Cache directory is typically `./cache_dir/` as initialized by `Cache("./cache_dir")` in `main.py` (though this line is commented out in `main.py` but present in other modules or was intended). `diskcache` handles its own internal configuration once the directory is set.

## 8. Setup, Installation, and Usage Instructions

(Based on `README.md` and insights from code)

**Prerequisites:**

1.  **Python:** Version 3.9 or higher.
2.  **Suricata IDS:**
    *   Must be installed and the `suricata` command must be in the system's PATH.
    *   Installation Guide: [https://suricata.readthedocs.io/en/latest/install.html](https://suricata.readthedocs.io/en/latest/install.html)
    *   Requires a configuration file (`suricata.yaml`) and rules. The project includes a `suricata.yaml` and a `Rules/` directory.
3.  **Zeek Network Security Monitor:**
    *   Must be installed and the `zeek` command must be in the system's PATH.
    *   Installation Guide: [https://docs.zeek.org/en/master/install.html](https://docs.zeek.org/en/master/install.html)
4.  **Mermaid CLI (`mmdc`):**
    *   Required for rendering Mermaid diagrams to PNG.
    *   Install via npm: `npm install -g @mermaid-js/mermaid-cli`
    *   This requires Node.js and npm to be installed.
5.  **Operating System:** Likely Linux or macOS, due to reliance on command-line Suricata/Zeek and typical paths. Windows Subsystem for Linux (WSL) might also work.

**Installation:**

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL
    cd CynsesAI-PCAP-Analyzer  # Or your cloned directory name
    ```

2.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (The `requirements.txt` file content was not provided, but it would typically include `langchain`, `langgraph`, `openai`, `requests`, `pandas`, `networkx`, `matplotlib`, `plotly`, `scapy`, `diskcache`, `transformers`, `torch`, etc.)

3.  **Configuration (`config/settings.py`):**
    *   Ensure you have a `config/settings.py` file. If an example file like `settings.py.example` is provided, copy or rename it.
    *   Update the following fields in `config/settings.py`:
        *   **`PCAP_FILE`**: Set the absolute path to your input PCAP file (e.g., `/path/to/your/GoldenEye.pcap`). The default in the provided file is an absolute path which might need changing.
        *   **`CHUTES_API_KEY`**: Your API key for Chutes AI (if using their service for LLM access).
        *   **`CHUTES_API_BASE`**: The API base URL for Chutes AI (default is likely correct).
        *   **`ABUSEIPDB_API_KEY`**: Your API key for AbuseIPDB.
        *   **`VIRUSTOTAL_API_KEY`**: Your API key for VirusTotal.
        *   **`SURICATA_CONFIG`**: Verify this path points to your `suricata.yaml` file. The project includes one at the root, so it might be `./suricata.yaml` or an absolute path if installed system-wide.
        *   **Paths for Outputs (Verify or Create):**
            *   `SURICATA_OUTPUT_DIR` (e.g., `"suricata_output/"`)
            *   `ZEEK_OUTPUT_DIR` (e.g., `"zeek_output/"`)
            *   `GRAPH_OUTPUT_DIR` (e.g., `"attack_graphs/"` or as defined in `settings.py`)
            *   The script creates these directories if they don't exist, but ensure write permissions.

**Usage:**

1.  **Place your PCAP file:**
    *   Make sure the `PCAP_FILE` variable in `config/settings.py` correctly points to the PCAP file you want to analyze.

2.  **Run the Analysis:**
    *   Navigate to the root directory of the project in your terminal.
    *   Execute the main script:
        ```bash
        python main.py
        ```
    *   The script will prompt you: "Do you want to run the network traffic classifier? (y/n):". Enter `y` or `n`.

3.  **Interactive Session (After Analysis):**
    *   Once the main analysis workflow is complete, an interactive session will start: "You can now ask questions about the analysis. Type 'exit' to quit."
    *   You can ask natural language questions about the findings, and the LLM will respond based on the summarized analysis state.

**Environment Variables:**
*   `USER_AGENT`: Set to "CynsesAI-PCAP-Analyzer/1.0" in `main.py`.
*   `TOKENIZERS_PARALLELISM`: Set to "false" in `main.py` to avoid warnings with Hugging Face tokenizers.

## 9. Explanation of Outputs

The CynsesAI PCAP Analyzer generates several outputs:

*   **Main Analysis Report:**
    *   **`pcap_analysis_report.md`**: This is the primary output. A comprehensive Markdown document containing:
        *   Executive Summary
        *   Technical Findings (including Suricata alerts, protocol issues, threat intelligence hits, anomalies)
        *   Attack Flow Analysis (potentially with an embedded Mermaid diagram source)
        *   Threat Assessment
        *   Recommended Actions
        *   Mitre ATT&CK Mapping (mentioned as a section, LLM-generated)
        *   Threat Classification details.
*   **Visualizations (typically in `attack_graphs/` or `GRAPH_OUTPUT_DIR`):**
    *   **`attack_flow.png`**: A PNG image rendered from a Mermaid diagram, showing the detected attack flow and relationships between entities.
    *   **`enterprise_attack_graph.png`** (or similar name like `attack_graph.png` as generated by `visualization_node` in `main.py`): A static PNG image generated by NetworkX and Matplotlib. It displays IP addresses as nodes (sized and colored by threat score) and connections/alerts as edges.
    *   **`interactive_attack_graph.html`**: An interactive HTML graph generated by Plotly. Users can zoom, pan, and hover over nodes and edges to get more details (IPs, threat scores, alert descriptions, connection details).
    *   **`security_summary.html`**: An HTML file (generated by `modules/visualization.py` if its main block is run, but not directly part of the LangGraph flow in `main.py` unless `visualization_node` is modified to include it). Contains summary charts like alert severity distribution, top alert categories, etc. *Correction*: `main.py`'s `visualization_node` *does* call functions that would produce these, but the `generate_summary_report` is not explicitly called within the LangGraph chain in `main.py`. The `README.md` mentions it as an output.
*   **Raw Tool Logs:**
    *   **`suricata_output/` directory:**
        *   `eve.json`: Detailed JSON-formatted log of all events generated by Suricata. This is the primary source for Suricata event data.
        *   `fast.log`: A simpler, text-based alert log from Suricata, useful for quick reviews and parsed by `anomaly_detection.py` and `visualization.py`.
        *   `stats.log`: Suricata performance statistics.
        *   `suricata.log`: General operational logs for Suricata.
    *   **`zeek_output/` directory:**
        *   Contains various plain-text log files generated by Zeek, separated by protocol or log type. Examples:
            *   `conn.log`: Detailed connection logs (TCP, UDP, ICMP).
            *   `dns.log`: DNS queries and responses.
            *   `http.log`: HTTP requests and replies.
            *   `ssl.log`: SSL/TLS handshake information.
            *   `files.log`: Information about files transferred over the network.
            *   `packet_filter.log`, `smtp.log`, `x509.log`, etc.
*   **Cache Directory:**
    *   **`cache_dir/`** (and/or `modules/cache_dir/`): Contains cached data from `diskcache`. This is not a direct user output but an operational artifact that speeds up subsequent runs.
*   **Console Output:**
    *   During execution, the script prints status messages, logs, summaries of detected non-normal traffic (if the classifier is run), and eventually the report summary to the console.
    *   The interactive GPT session also happens in the console.

## 10. Customization Points

The tool offers several areas for customization:

*   **Suricata Rules:**
    *   Modify existing rules or add new ones in the `Rules/` directory (e.g., `emerging-all.rules`, `suricata.rules`).
    *   Ensure the `suricata.yaml` file is updated to load the desired rule files.
    *   Custom rules can tailor threat detection to specific network environments or known indicators of compromise.
*   **Analysis Workflow (`main.py`):**
    *   The LangGraph workflow defined in `main.py` can be modified:
        *   **Add new nodes:** Integrate custom analysis scripts or tools as new nodes in the graph.
        *   **Remove nodes:** Disable certain analysis steps if they are not needed (e.g., skip anomaly detection).
        *   **Change node order/dependencies:** Restructure the flow, though dependencies must be respected.
        *   **Modify node logic:** Alter the functions executed by each node.
*   **LLM Prompts:**
    *   The prompts used for interacting with LLMs are embedded in various modules:
        *   `main.py` (for RAG, report structure, interactive session).
        *   `modules/protocol_analysis.py` (for analyzing Zeek logs).
        *   `modules/anomaly_detection.py` (for analyzing Suricata alert chunks).
        *   `modules/report_generation.py` (for the final report content).
    *   Adjusting these prompts can change the focus, depth, and format of the AI-generated analysis and reports.
*   **LLM Models and Parameters:**
    *   Switch between OpenAI/Chutes and Ollama by setting `USE_OLLAMA` in `config/settings.py`.
    *   Change the specific model names (e.g., `llama3.2`, `DeepSeek-V3-0324`) in `main.py` or respective modules.
    *   Adjust LLM parameters like `temperature`, `max_tokens` in the LLM initialization code within each module or `main.py`.
*   **Visualization Parameters (`modules/visualization.py`):**
    *   **Graph Layout:** Modify parameters in `nx.spring_layout` (e.g., `k`, `iterations`) to change graph appearance.
    *   **Colors, Sizes, Labels:** Adjust attributes for nodes and edges (e.g., `node_size`, `node_color`, `edge_color`, `width`) in `generate_networkx_plot` and `generate_plotly_interactive`.
    *   **Plotly Figure Layout:** Change titles, fonts, margins, legend appearance in `generate_plotly_interactive` and `generate_summary_report`.
*   **Threat Intelligence Sources:**
    *   Modify `modules/threat_intel.py` to:
        *   Integrate additional threat intelligence APIs.
        *   Change how data from existing APIs is processed or prioritized.
*   **Network Traffic Classifier (`modules/network_traffic_classifier.py`):**
    *   **Model:** Potentially replace the existing BERT model with a different pre-trained model or a custom-trained one. This would require changes to `AutoTokenizer.from_pretrained` and `AutoModelForSequenceClassification.from_pretrained`.
    *   **Feature Extraction:** Modify `processing_packet_conversion` if the new model expects different input features.
    *   **Classification Logic:** Adjust how predictions are handled or thresholded.
*   **Configuration (`config/settings.py`):**
    *   Easily change API keys, file paths, and other global settings.
*   **Suricata Configuration (`suricata.yaml`):**
    *   Advanced users can modify `suricata.yaml` to change Suricata's operational parameters, logging settings, rule paths, etc.

## 11. Contribution Guidelines and License

(Based on `README.md`)

**Contributing:**

Contributions to the CynsesAI PCAP Analyzer project are welcome. If you'd like to contribute:

1.  **Fork the Project:** Create your own fork of the repository.
2.  **Create your Feature Branch:**
    ```bash
    git checkout -b feature/AmazingFeature
    ```
3.  **Commit your Changes:** Make your modifications and commit them with clear messages:
    ```bash
    git commit -m 'Add some AmazingFeature'
    ```
4.  **Push to the Branch:**
    ```bash
    git push origin feature/AmazingFeature
    ```
5.  **Open a Pull Request:** Submit a pull request from your feature branch to the main project repository for review.

For major changes, it's recommended to open an issue first to discuss what you would like to change.

**License:**

The CynsesAI PCAP Analyzer is distributed under the **MIT License**.
See the `LICENSE` file (not provided in the input, but referenced in `README.md`) for more information. The MIT License is a permissive free software license, allowing for reuse within proprietary software provided all copies of the licensed software include a copy of the MIT License terms and the copyright notice.

---
This concludes the initial draft of the documentation.
