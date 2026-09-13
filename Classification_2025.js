// ============================================================
// CLASSIFICATION LULC - CAMPO - 2025 (Landsat 8 + Landsat 9)
// ============================================================

// 1. FILTRAGE EN AMONT
var l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(aoi);
var l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(aoi);
var l8l9 = l8.merge(l9);

var PALETTE = ['#1a6600', '#4daf4a', '#f5c518', '#00ced1', '#0033ff', '#d2691e'];
var NOMS_CLASSES = ['Forêt dense', 'Forêt secondaire', 'Agriculture / Jachère',
                     'Mangrove', 'Eau / Rivière', 'Sol nu / Bâti'];

function ajouterLegende(titre) {
  var legende = ui.Panel({style: {position: 'bottom-left', padding: '8px 15px'}});
  legende.add(ui.Label(titre, {fontWeight: 'bold', fontSize: '14px'}));
  for (var i = 0; i < PALETTE.length; i++) {
    var ligne = ui.Panel({layout: ui.Panel.Layout.flow('horizontal')});
    ligne.add(ui.Label('', {backgroundColor: PALETTE[i], padding: '8px', margin: '0 6px 2px 0'}));
    ligne.add(ui.Label(NOMS_CLASSES[i], {margin: '0 0 2px 0'}));
    legende.add(ligne);
  }
  Map.add(legende);
}

// ===== Préparation des images =====
function maskClouds(image) {
  var qa = image.select('QA_PIXEL');
  return image.updateMask(
    qa.bitwiseAnd(1 << 1).eq(0)
      .and(qa.bitwiseAnd(1 << 2).eq(0))
      .and(qa.bitwiseAnd(1 << 3).eq(0))
      .and(qa.bitwiseAnd(1 << 4).eq(0))
  );
}

function composite(collection, debut, fin) {
  return collection.filterDate(debut, fin)
    .map(maskClouds)
    .select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
            ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2'])
    .median()
    .multiply(0.0000275).add(-0.2);
}

var principal = composite(l8l9, '2025-01-01', '2025-12-31');
var secours   = composite(l8l9, '2023-01-01', '2026-06-30');
var image2025 = ee.ImageCollection([principal, secours]).mosaic().clip(aoi);

Map.centerObject(aoi, 10);
Map.addLayer(image2025, {bands: ['RED', 'GREEN', 'BLUE'], min: 0, max: 0.3}, 'Vraie couleur 2025');
ajouterLegende('LULC Campo 2025');

// ===== Indices spectraux =====
var NDVI  = image2025.normalizedDifference(['NIR', 'RED']).rename('NDVI');
var NDWI  = image2025.normalizedDifference(['GREEN', 'NIR']).rename('NDWI');
var NDBI  = image2025.normalizedDifference(['SWIR1', 'NIR']).rename('NDBI');
var MNDWI = image2025.normalizedDifference(['GREEN', 'SWIR1']).rename('MNDWI');

var imgClassif = image2025.addBands(NDVI).addBands(NDWI).addBands(NDBI).addBands(MNDWI);
var bandes = ['NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NDWI', 'NDBI', 'MNDWI'];

// ===== Zones d'entraînement (ROI) =====
var training = foret_dense.merge(foret_secondaire).merge(agri_jachere)
  .merge(Mangrove).merge(eau_riviere).merge(solnu_batis);

// ===== Classification supervisée  =====
var echantillons = imgClassif.select(bandes).sampleRegions({
  collection: training, 
  properties: ['class'], 
  scale: 30,
  tileScale: 4 
});

var dataset  = echantillons.randomColumn('alea', 42);
var trainSet = dataset.filter(ee.Filter.lessThan('alea', 0.8));
var testSet  = dataset.filter(ee.Filter.greaterThanOrEquals('alea', 0.8));

var classifieur = ee.Classifier.smileRandomForest(50).train({
  features: trainSet, classProperty: 'class', inputProperties: bandes
});

var classe2025 = imgClassif.select(bandes).classify(classifieur);
Map.addLayer(classe2025, {min: 1, max: 6, palette: PALETTE}, 'Classification LULC 2025');

// ===== Évaluation de la précision =====
var matrice = ee.ConfusionMatrix(testSet.classify(classifieur).errorMatrix('class', 'classification'));
print('Matrice de confusion 2025 :', matrice);
print('Précision globale 2025 :', matrice.accuracy());
print('Indice Kappa 2025 :', matrice.kappa());

// ===== Export GeoTIFF de la classification =====
Export.image.toDrive({
  image: classe2025,
  description: 'campo_LULC_2025',
  folder: 'campo_LULC',
  scale: 30,
  region: aoi,
  crs: 'EPSG:32632',
  maxPixels: 1e13
});

// ===== Export de la carte finale en couleur =====
Export.image.toDrive({
  image: classe2025.visualize({min: 1, max: 6, palette: PALETTE}),
  description: 'campo_carte_LULC_2025',
  folder: 'campo_LULC',
  scale: 30,
  region: aoi,
  crs: 'EPSG:32632',
  maxPixels: 1e13
});
