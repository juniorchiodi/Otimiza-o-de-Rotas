1. **Understand the problem**: The user wants to use a provided HTML template to generate a PDF for route planning, instead of the current FPDF-based generation.
2. **Review Codebase**:
   - `reports/pdf_builder.py` contains the `gerar_pdf_rota` function that currently uses `fpdf` to generate the PDF.
   - The user provided a new HTML template to be used.
3. **Changes**:
   - Change `reports/pdf_builder.py` to use `weasyprint` and `jinja2` to render the HTML template and save it to a PDF.
   - Create a new directory `templates` and save the provided HTML as `templates/route_template.html`.
   - Update `requirements.txt` to replace `fpdf2` with `weasyprint` and `jinja2`.
4. **Implementation details**:
   - I have already created the `templates/route_template.html` and modified it slightly (fixing `break-inside: avoid` syntax to `page-break-inside: avoid; break-inside: avoid;` to ensure components don't break across pages).
   - I have rewritten `reports/pdf_builder.py` to populate the `jinja2` template with the relevant variables, and then use `weasyprint` to render it to a PDF.
   - Tested the execution of `rota.py` using some dummy data and the PDF generation worked successfully.
   - I have modified `requirements.txt` to remove `fpdf2` and add `weasyprint` and `jinja2`.
5. **Pre-commit**: Complete the `pre_commit_instructions` checks.
6. **Submit**: Commit and push the changes.
