import json
import os

# --- Configuration ---
data_path = "model_output_golden_ranking/scores_calibrated.json"
output_filename = "Golden_report.html"
# ---------------------

try:
    with open(data_path, "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: Could not find {data_path}.")
    exit()

def generate_report(data, output_file):
    if not data:
        print("No data found.")
        return

    sorted_data = sorted(data, key=lambda x: x['calibrated_score'], reverse=True)

    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Quality Comparison Report - Large View</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: #f0f2f5; padding: 20px; color: #333; }
            h1 { text-align: center; font-weight: 300; margin-bottom: 5px;}
            
            .grid {
                display: grid;
                /* Increased min-width to 320px to force ~4 columns on standard screens */
                grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
                gap: 40px; 
                padding: 20px;
                max-width: 1600px;
                margin: 0 auto;
            }
            
            .card {
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                position: relative;
                transition: all 0.3s ease;
            }
            
            .card:hover {
                z-index: 100;
                box-shadow: 0 12px 30px rgba(0,0,0,0.1);
            }

            .img-container {
                width: 100%;
                /* Increased height to 300px for a larger visual impact */
                height: 300px;
                background: #e9ecef;
                border-radius: 12px 12px 0 0;
                position: relative;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .card img {
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
                border-radius: 12px 12px 0 0;
                transition: transform 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
                transform-origin: center center;
                position: relative;
                cursor: pointer;
            }
            
            /* The 1.5x Popout */
            .card:hover img {
                transform: scale(1.5);
                box-shadow: 0 30px 60px rgba(0,0,0,0.4);
            }
            
            .info { padding: 15px; border-top: 1px solid #eee;}
            .filename { 
                font-size: 0.75rem; 
                color: #777; 
                word-break: break-all; 
                display: block;
                margin-bottom: 8px;
            }
            .score-tag {
                display: inline-block;
                padding: 5px 12px;
                border-radius: 20px;
                color: white;
                font-weight: 700;
                font-size: 0.9rem;
                background-color: #28a745;
            }
        </style>
    </head>
    <body>
        <h1>Quality Comparison Report</h1>
        <p style="text-align: center; color: #666; font-size: 1rem;">Sorted by Score (Descending)</p>
        <div class="grid">
            {cards}
        </div>
    </body>
    </html>
    """

    cards_html = ""
    for item in sorted_data:
        card = f"""
        <div class="card">
            <div class="img-container">
                <img src="{item['path']}" alt="{item['filename']}">
            </div>
            <div class="info">
                <span class="filename">{item['filename']}</span>
                <div class="score-tag">Score: {item['calibrated_score']:.2f}</div>
            </div>
        </div>
        """
        cards_html += card

    with open(output_file, "w") as f:
        f.write(html_template.replace("{cards}", cards_html))

    print(f"Report generated: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    generate_report(data, output_filename)