# Publicacion en GitHub

## Recomendacion previa

Antes de publicar, revisar con tutor/consorcio si se pueden compartir las figuras y derivados de imagen incluidos. Si hay dudas, publicar primero el repositorio como privado.

## Crear el repositorio local

Desde la carpeta `repo_github_tfm`:

```bash
git init
git add .
git commit -m "Initial public TFM repository"
git branch -M main
```

## Subir a GitHub

Crear un repositorio vacio en GitHub y despues ejecutar:

```bash
git remote add origin https://github.com/USUARIO/NOMBRE_REPO.git
git push -u origin main
```

## Tamano

La version preparada pesa varios cientos de MB porque conserva resultados de imagen y arrays derivados. Ningun archivo deberia superar el limite duro de GitHub de 100 MB, pero el repositorio no es pequeno.

Si GitHub tarda mucho o se quiere una version mas ligera, las mejores opciones son:

- mover `*.npz` y `*.mha` a Git LFS;
- eliminar outputs intermedios y mantener solo los bloques finales;
- conservar los notebooks, scripts, CSV y figuras de memoria.

## Enlace en la memoria

Cuando el repositorio este subido, anadir el enlace en la memoria en una frase de este estilo:

> El codigo, notebooks y resultados derivados asociados a este TFM se encuentran disponibles en: `https://github.com/USUARIO/NOMBRE_REPO`.

