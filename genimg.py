import logging
import multiprocessing
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm

FONT_SIZE = 32
FONT = ImageFont.truetype("NotoSansCJKsc-Regular.otf", size=FONT_SIZE)
LOGGER = multiprocessing.get_logger()
FOLDER = "char_img"
CHUNKSIZE = 50

def gen_img(codepoint, font=FONT, folder=FOLDER):
    char = chr(int(codepoint, 16))
    test_img = Image.new("L", (FONT_SIZE, 4 * FONT_SIZE), 0)
    out_img = Image.new("L", (FONT_SIZE, FONT_SIZE), 0)
    d = ImageDraw.Draw(test_img)
    d.text((0, 0), char, fill=255, font=font)
    bbox = test_img.getbbox()
    d = ImageDraw.Draw(out_img)
    y = FONT_SIZE // 2 - (bbox[1] + bbox[3]) // 2
    if (bbox[3] - bbox[1]) > FONT_SIZE:
        LOGGER.warning(f"Character {char} at codepoint {codepoint} has height {bbox[3] - bbox[1]}")
    d.text((0, y), char, fill=255, font=font)
    out_img.save(f"{folder}/{codepoint}.png")

if __name__ == "__main__":
    multiprocessing.log_to_stderr().setLevel(logging.WARNING)
    with open("TGSCC-Unicode.txt", "r") as f:
        codepoints = [line.split()[1][2:] for line in f.readlines()[2:]]
    pool = multiprocessing.Pool()
    results = pool.imap_unordered(gen_img, codepoints, chunksize=CHUNKSIZE)
    for _ in tqdm(results, total=len(codepoints)):
        pass
    pool.close()
