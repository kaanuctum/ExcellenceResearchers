from os import environ
def set_proxy():
    environ["http_proxy"] = "proxy.biblio.tu-muenchen.de:8080"
    environ["https_proxy"] = "proxy.biblio.tu-muenchen.de:8080"

