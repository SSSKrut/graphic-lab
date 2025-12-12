# Path Tracer

Рендеринг сцены Cornell Box методом трассировки путей.

## Зависимости

- GLFW3
- GLM
- OpenGL
- Embree 4

## Сборка

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Запуск

```bash
./build/pathtracer
```

## Настройки

### Камера
- Position (X, Y, Z) — позиция камеры
- Look At (X, Y, Z) — точка, куда смотрит камера
- FOV — угол обзора

### Рендер
- SPP — количество сэмплов на пиксель
- Max Depth — глубина трассировки (отскоков)
- Gamma — гамма-коррекция (по умолчанию 2.2)
- Resolution — разрешение изображения (500-1000)

### Экспорт
- Save to PPM — сохранение результата в файл PPM

### Результат
![Рендер коробки Корнелла](github.com/SSSKrut/graphic-lab/blob/main/output/render.ppm)
