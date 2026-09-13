// ============================================================
// CLASSIFICATION LULC - CAMPO - 2015 (Landsat 8)
// Résolution 30 m | Projection EPSG:32632 (UTM 32N)
// ============================================================
// Imports requis : aoi, l8,
// forêt_dense, forêt_secondaire, agri_jachere, Mangrove,
// eau_rivière, solnu_batis (chacun avec propriété "class" 1 à 6)

// periode choisie 2014_2015
var l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")

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

// ===== 1. Préparation et affichage en vraie couleur =====
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
  return collection.filterBounds(aoi).filterDate(debut, fin)
    .map(maskClouds)
    .select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
            ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2'])
    .median()
    .multiply(0.0000275).add(-0.2);
}

// Composite cible (2014-2015) + composite de secours (2013-2017) pour combler les trous
var principal = composite(l8, '2014-01-01', '2015-12-31');
var secours   = composite(l8, '2013-01-01', '2017-12-31');
var image2015 = ee.ImageCollection([principal, secours]).mosaic().clip(aoi);

Map.centerObject(aoi, 10);
Map.addLayer(image2015, {bands: ['RED', 'GREEN', 'BLUE'], min: 0, max: 0.3}, 'Vraie couleur 2015');
ajouterLegende('LULC Campo 2015'); // légende visible dès maintenant, avant même la classification

// ===== 2. Vérification des trous (pixels sans donnée) =====
var trous = image2015.select('RED').mask().not().selfMask();
Map.addLayer(trous, {palette: ['red']}, 'Pixels manquants', false);

// ===== 3. Indices spectraux =====
var NDVI  = image2015.normalizedDifference(['NIR', 'RED']).rename('NDVI');
var NDWI  = image2015.normalizedDifference(['GREEN', 'NIR']).rename('NDWI');
var NDBI  = image2015.normalizedDifference(['SWIR1', 'NIR']).rename('NDBI');
var MNDWI = image2015.normalizedDifference(['GREEN', 'SWIR1']).rename('MNDWI');

Map.addLayer(NDVI, {min: 0, max: 0.9, palette: ['white', 'yellow', 'lightgreen', 'darkgreen']}, 'NDVI', false);
Map.addLayer(NDWI, {min: -0.5, max: 0.5, palette: ['brown', 'white', 'blue']}, 'NDWI', false);
Map.addLayer(NDBI, {min: -0.3, max: 0.3, palette: ['green', 'white', 'orange']}, 'NDBI', false);
Map.addLayer(MNDWI, {min: -0.5, max: 0.5, palette: ['brown', 'white', 'cyan']}, 'MNDWI', false);

// ===== 4. Couches préparées pour la classification =====
var imgClassif = image2015.addBands(NDVI).addBands(NDWI).addBands(NDBI).addBands(MNDWI);
var bandes = ['NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NDWI', 'NDBI', 'MNDWI'];

// ===== 5. Zones d'entraînement (ROI) =====
var training = foret_dense.merge(foret_secondaire).merge(agri_jachere)
  .merge(Mangrove).merge(eau_riviere).merge(solnu_batis);

// ===== 6. Classification supervisée (Random Forest) =====
var echantillons = imgClassif.select(bandes).sampleRegions({
  collection: training, properties: ['class'], scale: 30
});
var dataset  = echantillons.randomColumn('alea', 42);
var trainSet = dataset.filter(ee.Filter.lessThan('alea', 0.8));
var testSet  = dataset.filter(ee.Filter.greaterThanOrEquals('alea', 0.8));

var classifieur = ee.Classifier.smileRandomForest(100).train({
  features: trainSet, classProperty: 'class', inputProperties: bandes
});

var classe2015 = imgClassif.select(bandes).classify(classifieur);
Map.addLayer(classe2015, {min: 1, max: 6, palette: PALETTE}, 'Classification LULC 2015');

// ===== 7. Évaluation de la précision =====
var matrice = ee.ConfusionMatrix(testSet.classify(classifieur).errorMatrix('class', 'classification'));
print('Matrice de confusion 2015 :', matrice);
print('Précision globale 2015 :', matrice.accuracy());
print('Indice Kappa 2015 :', matrice.kappa());

// ===== 8. Export GeoTIFF de la classification =====
Export.image.toDrive({
  image: classe2015,
  description: 'campo_LULC_2015',
  folder: 'campo_LULC',
  scale: 30,
  region: aoi,
  crs: 'EPSG:32632',
  maxPixels: 1e13
});

// ===== 9. Export de la carte finale en couleur =====
Export.image.toDrive({
  image: classe2015.visualize({min: 1, max: 6, palette: PALETTE}),
  description: 'campo_carte_LULC_2015',
  folder: 'campo_LULC',
  scale: 30,
  region: aoi,
  crs: 'EPSG:32632',
  maxPixels: 1e13
});
