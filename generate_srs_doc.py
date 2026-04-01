import os
import zipfile
from datetime import date
from xml.sax.saxutils import escape

from PIL import Image


OUTPUT_FILE = "SRS_Stock_Price_Prediction_Saaransh_Malik.docx"
FALLBACK_OUTPUT_FILE = "SRS_Stock_Price_Prediction_Saaransh_Malik_with_graphs.docx"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGE_FILES = [
    (
        os.path.join(PROJECT_ROOT, "outputs", "matplotlib_graphs", "model_dashboard.png"),
        "Figure 1: Dashboard showing accuracy, error metrics, residual distribution, and percentile analysis.",
    ),
    (
        os.path.join(PROJECT_ROOT, "outputs", "matplotlib_graphs", "actual_vs_predicted.png"),
        "Figure 2: Comparison between actual closing price and predicted closing price on the test set.",
    ),
]


CONTENT = [
    ("title", "Software Requirements Specification"),
    ("subtitle", "Stock Price Prediction Project"),
    ("normal", "Prepared for: Saaransh Malik"),
    ("normal", "Programme: B.Tech, Computer Science and Engineering"),
    ("normal", "Institution: Manipal University Jaipur"),
    ("normal", "Registration Number: 23FE10CSE00781"),
    ("normal", f"Date: {date.today().strftime('%d %B %Y')}"),
    ("pagebreak", ""),
    ("heading1", "1. Introduction"),
    ("normal", "This Software Requirements Specification, or SRS, describes the stock price prediction project developed as an academic machine learning application. The main purpose of this document is to clearly explain what the system does, why it has been built, what kind of users may work with it, and what technical and functional requirements are needed for smooth operation. The project has been designed as a simple but meaningful example of how historical stock market data can be used to train a predictive model and generate easy-to-understand visual outputs."),
    ("normal", "The current implementation focuses on predicting stock closing prices by learning patterns from past records of the General Electric stock dataset. The system reads historical values such as open price, high price, low price, close price, and trading volume, prepares them for machine learning, trains a regression model, and then presents the results in an understandable way. Instead of keeping the project purely theoretical, the software has been made runnable on a student system and produces plotted graphs using matplotlib, which makes the final output more practical and presentable."),
    ("normal", "This document is written in a humanised academic style so that it remains formal enough for submission while still being easy to read. It can serve as a supporting report for project evaluation, viva preparation, documentation submission, or future project enhancement."),
    ("heading2", "1.1 Purpose of the System"),
    ("normal", "The purpose of the system is to demonstrate how machine learning can be applied to stock market data for the task of price prediction. The software is not intended to function as a real-time trading engine or a commercial financial recommendation platform. Instead, it is an educational and analytical project that helps a student understand the complete workflow of a predictive analytics application, starting from data loading and preprocessing and ending with model evaluation and visualization."),
    ("heading2", "1.2 Scope of the System"),
    ("normal", "The scope of this project includes data preprocessing, feature generation through time windows, model training using Gradient Boosting Regressor, evaluation with metrics such as R-squared, RMSE, and MAE, and graphical visualization of the prediction results. The scope does not include live stock exchange integration, online market streaming, portfolio management, user authentication, cloud deployment, or automated trading decisions. In short, the project is focused on learning, analysis, and presentation rather than production trading use."),
    ("heading2", "1.3 Intended Audience"),
    ("normal", "This SRS is intended for the student developer, faculty members, project evaluators, classmates, and any beginner who wants to understand the design of a stock prediction system. Since the document is written in simple and direct language, it can also help non-expert readers understand the project objectives and working flow without needing very deep knowledge of machine learning."),
    ("heading2", "1.4 Definitions and Abbreviations"),
    ("normal", "SRS means Software Requirements Specification. ML means Machine Learning. RMSE refers to Root Mean Squared Error. MAE means Mean Absolute Error. R-squared is a statistical score that indicates how well the model explains the variation in the target values. Matplotlib is the Python library used to generate graphs in this project. Gradient Boosting Regressor is the supervised learning algorithm currently used for prediction."),
    ("heading1", "2. Overall Description"),
    ("normal", "The stock price prediction system is a standalone desktop-based Python project. It does not require a web browser, server, or database engine to run. The software reads stock data from a text file, processes the data using pandas and scikit-learn, and then saves graph outputs as image files. The project is lightweight, easy to run on a normal student laptop, and suitable for demonstration in a lab or classroom setting."),
    ("normal", "Originally, the project structure contained references to an older deep learning approach based on LSTM and TensorFlow. However, the current runnable version uses scikit-learn's Gradient Boosting Regressor because it is simpler, more stable for the present environment, and easier to execute without heavy deep learning dependencies. This decision keeps the project practical while still preserving the core objective of stock price prediction."),
    ("heading2", "2.1 Product Perspective"),
    ("normal", "The project can be viewed as a compact analytical software product with three major layers. The first layer is the data layer, where the system reads and prepares historical stock information. The second layer is the model layer, where the machine learning algorithm learns patterns from the training data and predicts future close prices. The third layer is the presentation layer, where the outputs are shown through metrics and matplotlib-based plots. These layers are not separated into independent microservices or modules, but they are logically distinct and help explain the design clearly."),
    ("heading2", "2.2 Product Functions"),
    ("normal", "The main functions of the software are as follows. It loads the dataset from the project folder. It selects useful stock attributes such as Open, High, Low, Close, and Volume. It normalizes data values so that the model can learn more effectively. It converts time-series data into fixed-length supervised learning windows. It trains a prediction model using historical examples. It calculates performance metrics on training and testing data. It generates visual graphs showing accuracy, error values, residual distribution, and actual-versus-predicted trends. It stores output images in the outputs directory for later inspection."),
    ("heading2", "2.3 User Characteristics"),
    ("normal", "The expected primary user is a student with basic knowledge of Python and machine learning. The user does not need to be an expert software engineer, but should understand how to open a terminal, activate a virtual environment, and run a Python script. Secondary users may include instructors or reviewers who only need to inspect results and documentation rather than modify the code."),
    ("heading2", "2.4 Operating Environment"),
    ("normal", "The project runs in a Windows-based environment using PowerShell and Python in a virtual environment. The currently verified setup uses Python 3.11 in the project-specific .venv directory. Required software libraries include numpy, pandas, scikit-learn, and matplotlib. Since graph generation is file based, the project can work without a graphical desktop backend by using matplotlib's non-interactive Agg mode."),
    ("heading2", "2.5 Constraints"),
    ("normal", "The software depends on the availability of the input dataset file and the installed Python packages. It works on historical static data and cannot claim real-time market awareness. The model output depends on past patterns and therefore cannot guarantee future market correctness. Another practical constraint is that the project is intentionally simplified for academic use, so several real-world financial system requirements have been left outside the project scope."),
    ("heading2", "2.6 Assumptions and Dependencies"),
    ("normal", "The project assumes that the dataset file is clean, structured, and available in the expected project directory. It also assumes that the user has permission to create files in the outputs folder and can install required Python dependencies. Another assumption is that the performance values reported by the model are acceptable for an educational demonstration, even though they should not be interpreted as evidence of investment-grade forecasting."),
    ("heading1", "3. Specific Requirements"),
    ("heading2", "3.1 Functional Requirements"),
    ("normal", "FR-1: The system shall read stock market data from the provided input file."),
    ("normal", "FR-2: The system shall extract the columns Open, High, Low, Close, and Volume for model processing."),
    ("normal", "FR-3: The system shall split the dataset into training and testing parts using a fixed sequential split so that the time order remains meaningful."),
    ("normal", "FR-4: The system shall normalize feature values using MinMaxScaler before model training."),
    ("normal", "FR-5: The system shall generate supervised learning samples using a 60-day sliding window approach."),
    ("normal", "FR-6: The system shall train a Gradient Boosting Regressor model on the prepared feature set."),
    ("normal", "FR-7: The system shall produce predicted values for both training and testing data."),
    ("normal", "FR-8: The system shall compute model evaluation metrics including R-squared, RMSE, and MAE."),
    ("normal", "FR-9: The system shall generate matplotlib plots and save them as PNG files in the outputs folder."),
    ("normal", "FR-10: The system shall provide a runnable default entry point so that a user can execute the project with a single command."),
    ("heading2", "3.2 Non-Functional Requirements"),
    ("normal", "The software should be easy to run and easy to understand. It should finish execution within a reasonable time on a normal student laptop. The code should remain readable so that it can be used for learning and future enhancement. The graphs should be clear enough for report inclusion or viva explanation. The project should avoid unnecessary complexity and should use libraries that are commonly available in academic Python environments."),
    ("heading2", "3.3 Performance Requirements"),
    ("normal", "The software should complete model training and graph generation in a short duration under normal conditions. In the current environment, the simplified model completes within seconds rather than minutes, which makes it suitable for classroom demonstration. The system should also save the generated graphs without requiring manual intervention, so that the user can directly review the image outputs after execution."),
    ("heading2", "3.4 Reliability Requirements"),
    ("normal", "The project should consistently produce outputs when the required dependencies and input file are present. A reliable academic project does not need enterprise-level fault tolerance, but it should avoid avoidable runtime errors such as missing imports, broken paths, or unsupported visualization backends. The current runnable setup addresses these practical reliability needs."),
    ("heading2", "3.5 Usability Requirements"),
    ("normal", "Usability is important because the likely user is a student. For that reason, the project follows a simple structure and uses direct script names. The generated graphs help users interpret results visually rather than depending only on raw numerical output. This improves presentation quality and makes the system more useful for report writing, viva explanation, and academic demonstration."),
    ("heading2", "3.6 Security Requirements"),
    ("normal", "This project does not store personal user accounts, financial credentials, or confidential business records. Therefore, advanced security requirements such as encryption, role-based access control, and audit logs are not necessary in the current version. However, basic software safety still matters, which means the project should only read expected local files and write outputs within its own workspace."),
    ("heading1", "4. External Interface Requirements"),
    ("heading2", "4.1 User Interface"),
    ("normal", "The project uses a command-line interface. The user runs the Python script from the project directory. After execution, summary information is shown in the terminal and graphs are saved as image files. This is a simple but effective interface for an academic analytics project. If needed in the future, the project may be extended with a graphical user interface or a web dashboard."),
    ("heading2", "4.2 Hardware Interface"),
    ("normal", "The project does not need any specialized hardware. A regular laptop or desktop computer with sufficient memory to run Python is enough. No sensors, external devices, GPUs, or trading terminals are required for the current implementation."),
    ("heading2", "4.3 Software Interface"),
    ("normal", "The software interacts with Python packages including numpy for numerical operations, pandas for data handling, scikit-learn for preprocessing and model training, and matplotlib for graph plotting. It also interacts with the operating system file structure for reading the input dataset and writing the output plots."),
    ("heading2", "4.4 Communication Interface"),
    ("normal", "There is no active network communication in the project during normal execution. The software does not call external APIs or fetch live market data. This keeps the system simple, reproducible, and suitable for offline academic demonstrations."),
    ("heading1", "5. System Design Considerations"),
    ("normal", "The system follows a logical sequence that can be described in five major stages. First, it loads the stock dataset and selects useful columns. Second, it normalizes the selected values to make the learning process stable. Third, it creates time-window features so that each prediction is based on recent historical behaviour. Fourth, it trains the Gradient Boosting model and calculates prediction results. Fifth, it converts those results into clear matplotlib plots that can be stored and reviewed. This sequence reflects a practical and teachable machine learning workflow."),
    ("normal", "One of the strengths of this design is its simplicity. The code is compact enough for a student to understand without getting lost in deep framework complexity. At the same time, it still demonstrates important concepts like preprocessing, supervised learning, evaluation metrics, and data visualization. That balance makes the project suitable for educational submission."),
    ("heading1", "6. Results and Graphical Output"),
    ("normal", "The final system produces plotted visual outputs using matplotlib. These figures make the project easier to explain during viva, project demonstration, and report evaluation because they show the predictive performance of the model in a visual and immediate way. The dashboard graph summarizes multiple evaluation perspectives at once, while the actual-versus-predicted plot highlights how close the model is to the observed stock values over the test range."),
    ("normal", "Including these result images in the document makes the SRS more complete because it connects the written requirements to the implemented output. It also demonstrates that the system is not only designed on paper but is working in a practical form with measurable results."),
    ("images", ""),
    ("heading1", "7. Future Enhancements"),
    ("normal", "The project can be improved in many ways in the future. One enhancement would be the addition of technical indicators such as moving averages, RSI, MACD, or Bollinger Bands to provide richer model features. Another enhancement would be using multiple stocks instead of a single dataset so that the software becomes more general. The system may also be extended with a GUI or web dashboard where the user can upload a dataset and instantly view graphs. More advanced model choices, such as XGBoost, LSTM, or hybrid approaches, may also be explored if dependency support and project scope allow."),
    ("normal", "Another valuable future improvement would be better evaluation design through walk-forward validation instead of a single split. This would make the model assessment more realistic for time-series data. Further improvements could include prediction for several future days, export of a PDF report, or automated comparison among multiple algorithms."),
    ("heading1", "8. Conclusion"),
    ("normal", "In conclusion, the Stock Price Prediction project is a well-defined academic software application that demonstrates the use of machine learning in financial data analysis. The project accepts historical stock data, prepares it through preprocessing, trains a predictive model, evaluates the result using accepted metrics, and presents the outcome using matplotlib graphs. Its current implementation is lightweight, runnable, and suitable for submission, demonstration, and learning."),
    ("normal", "From a student project perspective, the software succeeds because it is practical, understandable, and visually demonstrative. It does not try to solve every real-world financial problem, but it does successfully show how data science concepts can be translated into working software. This makes it an appropriate and meaningful B.Tech CSE project for Saaransh Malik at Manipal University Jaipur."),
]


def make_paragraph(text, style=None, align=None, page_break=False):
    if page_break:
        return "<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>"
    ppr = ""
    if style or align:
        parts = []
        if style:
            parts.append(f"<w:pStyle w:val=\"{style}\"/>")
        if align:
            parts.append(f"<w:jc w:val=\"{align}\"/>")
        ppr = f"<w:pPr>{''.join(parts)}</w:pPr>"
    return f"<w:p>{ppr}<w:r><w:t xml:space=\"preserve\">{escape(text)}</w:t></w:r></w:p>"


def image_dimensions(path, max_width_inches=6.0):
    with Image.open(path) as img:
        width_px, height_px = img.size
    max_width_emu = int(max_width_inches * 914400)
    width_emu = width_px * 9525
    height_emu = height_px * 9525
    if width_emu > max_width_emu:
        ratio = max_width_emu / width_emu
        width_emu = int(width_emu * ratio)
        height_emu = int(height_emu * ratio)
    return width_emu, height_emu


def make_image_paragraph(rel_id, image_index, name, width_emu, height_emu):
    return f"""
<w:p>
  <w:pPr><w:jc w:val="center"/></w:pPr>
  <w:r>
    <w:drawing>
      <wp:inline distT="0" distB="0" distL="0" distR="0">
        <wp:extent cx="{width_emu}" cy="{height_emu}"/>
        <wp:effectExtent l="0" t="0" r="0" b="0"/>
        <wp:docPr id="{image_index}" name="{escape(name)}"/>
        <wp:cNvGraphicFramePr>
          <a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/>
        </wp:cNvGraphicFramePr>
        <a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
          <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
            <pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
              <pic:nvPicPr>
                <pic:cNvPr id="0" name="{escape(name)}"/>
                <pic:cNvPicPr/>
              </pic:nvPicPr>
              <pic:blipFill>
                <a:blip r:embed="{rel_id}"/>
                <a:stretch><a:fillRect/></a:stretch>
              </pic:blipFill>
              <pic:spPr>
                <a:xfrm>
                  <a:off x="0" y="0"/>
                  <a:ext cx="{width_emu}" cy="{height_emu}"/>
                </a:xfrm>
                <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
              </pic:spPr>
            </pic:pic>
          </a:graphicData>
        </a:graphic>
      </wp:inline>
    </w:drawing>
  </w:r>
</w:p>
"""


def build_document_xml():
    paragraphs = []
    image_counter = 1
    rel_counter = 100

    for kind, text in CONTENT:
        if kind == "title":
            paragraphs.append(make_paragraph(text, style="Title", align="center"))
        elif kind == "subtitle":
            paragraphs.append(make_paragraph(text, style="Subtitle", align="center"))
        elif kind == "heading1":
            paragraphs.append(make_paragraph(text, style="Heading1"))
        elif kind == "heading2":
            paragraphs.append(make_paragraph(text, style="Heading2"))
        elif kind == "pagebreak":
            paragraphs.append(make_paragraph("", page_break=True))
        elif kind == "images":
            for path, caption in IMAGE_FILES:
                rel_id = f"rId{rel_counter}"
                rel_counter += 1
                width_emu, height_emu = image_dimensions(path)
                paragraphs.append(make_image_paragraph(rel_id, image_counter, os.path.basename(path), width_emu, height_emu))
                paragraphs.append(make_paragraph(caption, align="center"))
                image_counter += 1
        else:
            paragraphs.append(make_paragraph(text))

    body = "".join(paragraphs)
    sect = (
        "<w:sectPr>"
        "<w:pgSz w:w=\"12240\" w:h=\"15840\"/>"
        "<w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" w:left=\"1440\" "
        "w:header=\"708\" w:footer=\"708\" w:gutter=\"0\"/>"
        "</w:sectPr>"
    )
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<w:document xmlns:wpc=\"http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas\" "
        "xmlns:mc=\"http://schemas.openxmlformats.org/markup-compatibility/2006\" "
        "xmlns:o=\"urn:schemas-microsoft-com:office:office\" "
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" "
        "xmlns:m=\"http://schemas.openxmlformats.org/officeDocument/2006/math\" "
        "xmlns:v=\"urn:schemas-microsoft-com:vml\" "
        "xmlns:wp14=\"http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing\" "
        "xmlns:wp=\"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing\" "
        "xmlns:w10=\"urn:schemas-microsoft-com:office:word\" "
        "xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" "
        "xmlns:w14=\"http://schemas.microsoft.com/office/word/2010/wordml\" "
        "xmlns:wpg=\"http://schemas.microsoft.com/office/word/2010/wordprocessingGroup\" "
        "xmlns:wpi=\"http://schemas.microsoft.com/office/word/2010/wordprocessingInk\" "
        "xmlns:wne=\"http://schemas.microsoft.com/office/word/2006/wordml\" "
        "xmlns:wps=\"http://schemas.microsoft.com/office/word/2010/wordprocessingShape\" "
        "mc:Ignorable=\"w14 wp14\">"
        f"<w:body>{body}{sect}</w:body></w:document>"
    )


def build_styles_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>
        <w:sz w:val="24"/>
        <w:szCs w:val="24"/>
      </w:rPr>
    </w:rPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:after="160" w:line="360" w:lineRule="auto"/>
    </w:pPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:before="200" w:after="160"/>
    </w:pPr>
    <w:rPr>
      <w:b/>
      <w:sz w:val="34"/>
      <w:szCs w:val="34"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Subtitle">
    <w:name w:val="Subtitle"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:rPr>
      <w:i/>
      <w:sz w:val="28"/>
      <w:szCs w:val="28"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:before="220" w:after="120"/>
    </w:pPr>
    <w:rPr>
      <w:b/>
      <w:sz w:val="28"/>
      <w:szCs w:val="28"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:before="160" w:after="80"/>
    </w:pPr>
    <w:rPr>
      <w:b/>
      <w:sz w:val="24"/>
      <w:szCs w:val="24"/>
    </w:rPr>
  </w:style>
</w:styles>
"""


def build_core_xml():
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Software Requirements Specification - Stock Price Prediction</dc:title>
  <dc:subject>Academic Project Documentation</dc:subject>
  <dc:creator>Codex</dc:creator>
  <cp:keywords>SRS, Stock Price Prediction, Machine Learning, Saaransh Malik</cp:keywords>
  <dc:description>Software Requirements Specification for Stock Price Prediction project with result images.</dc:description>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{date.today().isoformat()}T00:00:00Z</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{date.today().isoformat()}T00:00:00Z</dcterms:modified>
</cp:coreProperties>
"""


def build_app_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Office Word</Application>
  <DocSecurity>0</DocSecurity>
  <ScaleCrop>false</ScaleCrop>
  <HeadingPairs>
    <vt:vector size="2" baseType="variant">
      <vt:variant><vt:lpstr>Title</vt:lpstr></vt:variant>
      <vt:variant><vt:i4>1</vt:i4></vt:variant>
    </vt:vector>
  </HeadingPairs>
  <TitlesOfParts>
    <vt:vector size="1" baseType="lpstr">
      <vt:lpstr>Document</vt:lpstr>
    </vt:vector>
  </TitlesOfParts>
  <Company>Manipal University Jaipur</Company>
  <LinksUpToDate>false</LinksUpToDate>
  <SharedDoc>false</SharedDoc>
  <HyperlinksChanged>false</HyperlinksChanged>
  <AppVersion>16.0000</AppVersion>
</Properties>
"""


def build_document_relationships():
    rels = [
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>",
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">",
    ]
    rel_id = 100
    media_index = 1
    for image_path, _ in IMAGE_FILES:
        rels.append(
            f"<Relationship Id=\"rId{rel_id}\" "
            "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/image\" "
            f"Target=\"media/image{media_index}.png\"/>"
        )
        rel_id += 1
        media_index += 1
    rels.append("</Relationships>")
    return "".join(rels)


def write_docx(path):
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types)
        docx.writestr("_rels/.rels", rels)
        docx.writestr("docProps/core.xml", build_core_xml())
        docx.writestr("docProps/app.xml", build_app_xml())
        docx.writestr("word/document.xml", build_document_xml())
        docx.writestr("word/styles.xml", build_styles_xml())
        docx.writestr("word/_rels/document.xml.rels", build_document_relationships())

        for index, (image_path, _) in enumerate(IMAGE_FILES, start=1):
            with open(image_path, "rb") as image_file:
                docx.writestr(f"word/media/image{index}.png", image_file.read())


if __name__ == "__main__":
    target = os.path.join(PROJECT_ROOT, OUTPUT_FILE)
    try:
        write_docx(target)
        print(target)
    except PermissionError:
        fallback_target = os.path.join(PROJECT_ROOT, FALLBACK_OUTPUT_FILE)
        write_docx(fallback_target)
        print(fallback_target)
