from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "Fotos_Memoria" / "pipeline_metodologia_registro.png"


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


TITLE_FONT = load_font(48, bold=True)
SECTION_FONT = load_font(30, bold=True)
BODY_FONT = load_font(25)
SMALL_FONT = load_font(21)


def centered_text(draw: ImageDraw.ImageDraw, box, text: str, font, fill=(28, 32, 36)) -> None:
    x1, y1, x2, y2 = box
    bounds = draw.multiline_textbbox((0, 0), text, font=font, align="center", spacing=5)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.multiline_text(
        ((x1 + x2 - width) / 2, (y1 + y2 - height) / 2 - bounds[1]),
        text,
        font=font,
        fill=fill,
        align="center",
        spacing=5,
    )


def box(draw, xy, title: str, subtitle: str, fill, outline) -> None:
    draw.rounded_rectangle(xy, radius=8, fill=fill, outline=outline, width=3)
    x1, y1, x2, y2 = xy
    centered_text(draw, (x1 + 14, y1 + 8, x2 - 14, y1 + 58), title, SECTION_FONT)
    centered_text(draw, (x1 + 18, y1 + 56, x2 - 18, y2 - 8), subtitle, BODY_FONT, fill=(45, 49, 53))


def arrow(draw, start, end, color=(80, 86, 92), width=5) -> None:
    draw.line((start, end), fill=color, width=width)
    x2, y2 = end
    x1, y1 = start
    if abs(x2 - x1) >= abs(y2 - y1):
        direction = 1 if x2 > x1 else -1
        points = [(x2, y2), (x2 - direction * 18, y2 - 11), (x2 - direction * 18, y2 + 11)]
    else:
        direction = 1 if y2 > y1 else -1
        points = [(x2, y2), (x2 - 11, y2 - direction * 18), (x2 + 11, y2 - direction * 18)]
    draw.polygon(points, fill=color)


canvas = Image.new("RGB", (2400, 1500), "white")
draw = ImageDraw.Draw(canvas)

centered_text(draw, (90, 30, 2310, 105), "Pipeline metodológico del registro H&E–HSI", TITLE_FONT)

# Entradas y preprocesado específico por modalidad.
box(
    draw,
    (110, 145, 1090, 300),
    "Imagen histológica H&E",
    "Archivo MRXS + anotaciones XML\nSegmentación del tejido y extracción del contorno",
    (252, 235, 245),
    (177, 74, 137),
)
box(
    draw,
    (1310, 145, 2290, 300),
    "Imagen hiperespectral HSI",
    "RAW → NRM_EDU → pseudo-RGB\nDetección de la caja y segmentación del espécimen",
    (226, 246, 244),
    (32, 128, 117),
)

arrow(draw, (600, 300), (600, 365))
arrow(draw, (1800, 300), (1800, 365))

box(
    draw,
    (360, 365, 2040, 505),
    "Representaciones comparables",
    "Homogeneización de la escala física · máscaras binarias · contornos · mapas de distancia con signo",
    (242, 244, 246),
    (91, 101, 111),
)

arrow(draw, (1200, 505), (1200, 570))

# Dos familias metodológicas.
box(
    draw,
    (110, 570, 1125, 720),
    "Baseline clásico",
    "Rígido por máscaras → afín mediante ECC → deformable TV-L1\nSelección jerárquica y mecanismo de retroceso",
    (230, 239, 250),
    (52, 94, 149),
)
box(
    draw,
    (1275, 570, 2290, 720),
    "Registro mediante aprendizaje profundo",
    "Optimización independiente por espécimen\nVoxelMorph-style · TransMorph-style · DeeperHistReg exploratorio",
    (237, 246, 231),
    (74, 126, 62),
)

arrow(draw, (620, 720), (620, 800))
arrow(draw, (1780, 720), (1780, 800))

box(
    draw,
    (110, 800, 1125, 955),
    "Salidas del baseline",
    "H&E registrada · máscara transformada · matriz o campo denso\nMétricas, superposiciones y banderas de revisión",
    (235, 242, 250),
    (52, 94, 149),
)
box(
    draw,
    (1275, 800, 2290, 955),
    "Salidas de los modelos neuronales",
    "Campo de desplazamiento y H&E deformada\nHistorial de optimización y calidad geométrica",
    (241, 247, 235),
    (74, 126, 62),
)

arrow(draw, (620, 955), (620, 1030))
arrow(draw, (1780, 955), (1780, 1030))

box(
    draw,
    (360, 1030, 2040, 1195),
    "Protocolo común de evaluación",
    "Dice e IoU · distancia entre contornos · magnitud del desplazamiento\nJacobiano del campo · inspección visual de contornos y superposiciones",
    (252, 245, 224),
    (161, 114, 29),
)

arrow(draw, (1200, 1195), (1200, 1275))

box(
    draw,
    (615, 1275, 1785, 1415),
    "Resultado registrado y trazable",
    "H&E transformada en el sistema de coordenadas de la HSI\ncon parámetros, métricas y archivos de salida reproducibles",
    (238, 239, 244),
    (72, 74, 92),
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
canvas.save(OUTPUT_PATH, dpi=(300, 300))
print(f"Figura guardada en: {OUTPUT_PATH}")
