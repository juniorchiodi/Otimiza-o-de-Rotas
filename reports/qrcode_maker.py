from utils.console import print_colorido, Fore
import os

try:
    import qrcode
    QRCODE_AVAILABLE = True
except ImportError:
    QRCODE_AVAILABLE = False

def gerar_qrcodes_rota(coordenadas, ordem_rota):
    arquivos_qr_gerados = []
    if QRCODE_AVAILABLE:
        print_colorido("📱 Gerando QR Codes para o motorista...", Fore.CYAN)
        chunk_size = 10
        for i in range(0, len(ordem_rota), chunk_size - 1):
            chunk = ordem_rota[i:i+chunk_size]
            if len(chunk) < 2: break

            orig_str = f"{coordenadas[chunk[0]][0]},{coordenadas[chunk[0]][1]}"
            dest_str = f"{coordenadas[chunk[-1]][0]},{coordenadas[chunk[-1]][1]}"
            url_maps = f"https://www.google.com/maps/dir/?api=1&origin={orig_str}&destination={dest_str}"

            if len(chunk) > 2:
                wp_str = "|".join([f"{coordenadas[w][0]},{coordenadas[w][1]}" for w in chunk[1:-1]])
                url_maps += f"&waypoints={wp_str}"

            qr = qrcode.make(url_maps)
            nome_qr = f"temp_qr_part_{i}.png"
            qr.save(nome_qr)
            arquivos_qr_gerados.append(nome_qr)
    return arquivos_qr_gerados

def apagar_qrcodes(arquivos_qr_gerados):
    for temp_img in arquivos_qr_gerados:
        try: os.remove(temp_img)
        except: pass
