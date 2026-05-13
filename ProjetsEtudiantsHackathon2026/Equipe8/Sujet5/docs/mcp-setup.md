# MCP servers utiles pour le Sujet 5

> Les configs MCP contiennent des chemins absolus spécifiques à la machine → on ne committe **pas** de `.mcp.json` cassé. Voici ce qu'il faut configurer (dans le `.mcp.json` à la racine du repo de travail, ou au niveau utilisateur via `claude mcp add`).

## `shom-wrecks` — anti-faux-positifs (épaves SHOM, eaux françaises)
Utilisé en Piste B pour écarter les détections qui coïncident avec une épave connue (Toulon, Brest…). Le serveur n'est **pas** sur npm — il faut le builder à la main :

```bash
git clone <repo mcp-shom-wrecks>   # cf. notes équipe
cd mcp-shom-wrecks && npm install && npm run build
# puis dans .mcp.json :
{
  "mcpServers": {
    "shom-wrecks": {
      "command": "node",
      "args": ["/chemin/absolu/vers/mcp-shom-wrecks/dist/index.js"]
    }
  }
}
```

Alternative sans MCP : interroger directement le **WFS épaves du SHOM** (service public, Licence Ouverte) — `epaves` layer sur `services.data.shom.fr`.

## `arxiv` — recherche bibliographique (optionnel)
Pour retrouver les papiers de référence (détection SAR, RF fingerprinting…). Installer `arxiv-mcp-server` (pip/uv tool) puis :

```json
{
  "mcpServers": {
    "arxiv": {
      "command": "arxiv-mcp-server",
      "args": ["--storage-path", "~/.arxiv-mcp-server/papers"]
    }
  }
}
```

La biblio déjà sélectionnée est dans `BDD-MinArm/hackathon-2026/docs/recherche-arxiv-sujets-3-5.md` — l'MCP arxiv n'est utile que pour aller plus loin.

## `github` (HTTP, portable)
```json
{
  "mcpServers": {
    "github": { "type": "http", "url": "https://api.githubcopilot.com/mcp/" }
  }
}
```

## Données satellite : pas besoin de MCP
On utilise des libs Python directement (`pystac-client` + `planetary-computer` pour le STAC Sentinel-1/2 — pas d'inscription requise ; `sentinelhub` ou Copernicus Data Space avec compte). Cf. `src/hunt.py`.
