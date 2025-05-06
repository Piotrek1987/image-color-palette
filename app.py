import os
from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import webcolors
from io import BytesIO

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

css3_names_to_hex = {
    'AliceBlue': '#f0f8ff', 'AntiqueWhite': '#faebd7', 'Aqua': '#00ffff', 'Aquamarine': '#7fffd4',
    'Azure': '#f0ffff', 'Beige': '#f5f5dc', 'Bisque': '#ffe4c4', 'Black': '#000000',
    'BlanchedAlmond': '#ffebcd', 'Blue': '#0000ff', 'BlueViolet': '#8a2be2', 'Brown': '#a52a2a',
    'BurlyWood': '#deb887', 'CadetBlue': '#5f9ea0', 'Chartreuse': '#7fff00', 'Chocolate': '#d2691e',
    'Coral': '#ff7f50', 'CornflowerBlue': '#6495ed', 'Cornsilk': '#fff8dc', 'Crimson': '#dc143c',
    'Cyan': '#00ffff', 'DarkBlue': '#00008b', 'DarkCyan': '#008b8b', 'DarkGoldenRod': '#b8860b',
    'DarkGray': '#a9a9a9', 'DarkGreen': '#006400', 'DarkKhaki': '#bdb76b', 'DarkMagenta': '#8b008b',
    'DarkOliveGreen': '#556b2f', 'DarkOrange': '#ff8c00', 'DarkOrchid': '#9932cc', 'DarkRed': '#8b0000',
    'DarkSalmon': '#e9967a', 'DarkSeaGreen': '#8fbc8f', 'DarkSlateBlue': '#483d8b', 'DarkSlateGray': '#2f4f4f',
    'DarkTurquoise': '#00ced1', 'DarkViolet': '#9400d3', 'DeepPink': '#ff1493', 'DeepSkyBlue': '#00bfff',
    'DimGray': '#696969', 'DodgerBlue': '#1e90ff', 'FireBrick': '#b22222', 'FloralWhite': '#fffaf0',
    'ForestGreen': '#228b22', 'Fuchsia': '#ff00ff', 'Gainsboro': '#dcdcdc', 'GhostWhite': '#f8f8ff',
    'Gold': '#ffd700', 'GoldenRod': '#daa520', 'Gray': '#808080', 'Green': '#008000',
    'GreenYellow': '#adff2f', 'HoneyDew': '#f0fff0', 'HotPink': '#ff69b4', 'IndianRed': '#cd5c5c',
    'Indigo': '#4b0082', 'Ivory': '#fffff0', 'Khaki': '#f0e68c', 'Lavender': '#e6e6fa',
    'LavenderBlush': '#fff0f5', 'LawnGreen': '#7cfc00', 'LemonChiffon': '#fffacd', 'LightBlue': '#add8e6',
    'LightCoral': '#f08080', 'LightCyan': '#e0ffff', 'LightGoldenRodYellow': '#fafad2',
    'LightGray': '#d3d3d3', 'LightGreen': '#90ee90', 'LightPink': '#ffb6c1', 'LightSalmon': '#ffa07a',
    'LightSeaGreen': '#20b2aa', 'LightSkyBlue': '#87cefa', 'LightSlateGray': '#778899',
    'LightSteelBlue': '#b0c4de', 'LightYellow': '#ffffe0', 'Lime': '#00ff00', 'LimeGreen': '#32cd32',
    'Linen': '#faf0e6', 'Magenta': '#ff00ff', 'Maroon': '#800000', 'MediumAquaMarine': '#66cdaa',
    'MediumBlue': '#0000cd', 'MediumOrchid': '#ba55d3', 'MediumPurple': '#9370db', 'MediumSeaGreen': '#3cb371',
    'MediumSlateBlue': '#7b68ee', 'MediumSpringGreen': '#00fa9a', 'MediumTurquoise': '#48d1cc',
    'MediumVioletRed': '#c71585', 'MidnightBlue': '#191970', 'MintCream': '#f5fffa', 'MistyRose': '#ffe4e1',
    'Moccasin': '#ffe4b5', 'NavajoWhite': '#ffdead', 'Navy': '#000080', 'OldLace': '#fdf5e6',
    'Olive': '#808000', 'OliveDrab': '#6b8e23', 'Orange': '#ffa500', 'OrangeRed': '#ff4500',
    'Orchid': '#da70d6', 'PaleGoldenRod': '#eee8aa', 'PaleGreen': '#98fb98', 'PaleTurquoise': '#afeeee',
    'PaleVioletRed': '#db7093', 'PapayaWhip': '#ffefd5', 'PeachPuff': '#ffdab9', 'Peru': '#cd853f',
    'Pink': '#ffc0cb', 'Plum': '#dda0dd', 'PowderBlue': '#b0e0e6', 'Purple': '#800080',
    'RebeccaPurple': '#663399', 'Red': '#ff0000', 'RosyBrown': '#bc8f8f', 'RoyalBlue': '#4169e1',
    'SaddleBrown': '#8b4513', 'Salmon': '#fa8072', 'SandyBrown': '#f4a460', 'SeaGreen': '#2e8b57',
    'SeaShell': '#fff5ee', 'Sienna': '#a0522d', 'Silver': '#c0c0c0', 'SkyBlue': '#87ceeb',
    'SlateBlue': '#6a5acd', 'SlateGray': '#708090', 'Snow': '#fffafa', 'SpringGreen': '#00ff7f',
    'SteelBlue': '#4682b4', 'Tan': '#d2b48c', 'Teal': '#008080', 'Thistle': '#d8bfd8',
    'Tomato': '#ff6347', 'Turquoise': '#40e0d0', 'Violet': '#ee82ee', 'Wheat': '#f5deb3',
    'White': '#ffffff', 'WhiteSmoke': '#f5f5f5', 'Yellow': '#ffff00', 'YellowGreen': '#9acd32'
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_colors(image_path, num_colors=10):
    image = Image.open(image_path)
    image = image.resize((150, 150))
    image_np = np.array(image)
    pixels = image_np.reshape(-1, 3)

    kmeans = KMeans(n_clusters=num_colors, n_init=10)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_.astype(int)
    hex_colors = ['#{:02x}{:02x}{:02x}'.format(*color) for color in colors]
    named_colors = [get_color_name(h) for h in hex_colors]
    return list(zip(hex_colors, named_colors))



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        web_path = file_path.replace("\\", "/")
        colors = extract_colors(file_path)
        return render_template('result.html', image_url=web_path, colors=colors)

    return render_template('index.html')

def closest_color(requested_color):
    min_colors = {}
    for name, hex_code in css3_names_to_hex.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(hex_code)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]



def get_color_name(hex_code):
    try:
        return webcolors.hex_to_name(hex_code)
    except ValueError:
        rgb = webcolors.hex_to_rgb(hex_code)
        return closest_color(rgb)


@app.route("/download_txt")
def download_txt():
    hex_codes = request.args.getlist("hex")
    content = "\n".join(hex_codes)

    buffer = BytesIO()
    buffer.write(content.encode("utf-8"))
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="palette.txt", mimetype="text/plain")


@app.route("/download_css")
def download_css():
    hex_codes = request.args.getlist("hex")
    css_content = ""

    for i, hex_code in enumerate(hex_codes):
        css_content += f".color-{i + 1} {{ background-color: {hex_code}; }}\n"

    buffer = BytesIO()
    buffer.write(css_content.encode("utf-8"))
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="palette.css", mimetype="text/css")


if __name__ == '__main__':
    app.run(debug=True)
