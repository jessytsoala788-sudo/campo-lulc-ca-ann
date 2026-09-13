
// ============================================================
// CLASSIFICATION LULC - CAMPO - 2000 (Landsat 5 + Landsat 7)
// Résolution 30 m | Projection EPSG:32632 (UTM 32N)
// ============================================================
// Imports requis : aoi
// Crée dans Imports (avec propriété "class" 1 à 6) :
// forêt_dense_2000, forêt_secondaire_2000, agri_jachere_2000,
// Mangrove_2000, eau_rivière_2000, solnu_batis_2000

// Période choisie avant avril 2003 pour éviter le défaut
// SLC-off de Landsat 7 (bandes noires manquantes).

var l5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2');
var l7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2');
var l5l7 = l5.merge(l7);

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
    .select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'],
            ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2'])
    .median()
    .multiply(0.0000275).add(-0.2);
}

// Composite cible (1999-2001) + composite de secours (1998 à avril 2003) pour combler les trous
var principal = composite(l5l7, '1999-01-01', '2001-12-31');
var secours   = composite(l5l7, '1998-01-01', '2003-04-01');
var image2000 = ee.ImageCollection([principal, secours]).mosaic().clip(aoi);

Map.centerObject(aoi, 10);
Map.addLayer(image2000, {bands: ['RED', 'GREEN', 'BLUE'], min: 0, max: 0.3}, 'Vraie couleur 2000');
ajouterLegende('LULC Campo 2000'); // légende visible dès maintenant, avant même la classification

// ===== 2. Vérification des trous (pixels sans donnée) =====
var trous = image2000.select('RED').mask().not().selfMask();
Map.addLayer(trous, {palette: ['red']}, 'Pixels manquants', false);

// ===== 3. Indices spectraux =====
var NDVI  = image2000.normalizedDifference(['NIR', 'RED']).rename('NDVI');
var NDWI  = image2000.normalizedDifference(['GREEN', 'NIR']).rename('NDWI');
var NDBI  = image2000.normalizedDifference(['SWIR1', 'NIR']).rename('NDBI');
var MNDWI = image2000.normalizedDifference(['GREEN', 'SWIR1']).rename('MNDWI');

Map.addLayer(NDVI, {min: 0, max: 0.9, palette: ['white', 'yellow', 'lightgreen', 'darkgreen']}, 'NDVI', false);
Map.addLayer(NDWI, {min: -0.5, max: 0.5, palette: ['brown', 'white', 'blue']}, 'NDWI', false);
Map.addLayer(NDBI, {min: -0.3, max: 0.3, palette: ['green', 'white', 'orange']}, 'NDBI', false);
Map.addLayer(MNDWI, {min: -0.5, max: 0.5, palette: ['brown', 'white', 'cyan']}, 'MNDWI', false);

// ===== 4. Couches préparées pour la classification =====
var imgClassif = image2000.addBands(NDVI).addBands(NDWI).addBands(NDBI).addBands(MNDWI);
var bandes = ['NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NDWI', 'NDBI', 'MNDWI'];

// ===== 5. Zones d'entraînement (ROI) =====
var training = foret_dense_2000.merge(foret_secondaire_2000).merge(agri_jachere_2000)
  .merge(Mangrove_2000).merge(eau_riviere_2000).merge(solnu_batis_2000);

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

var classe2000 = imgClassif.select(bandes).classify(classifieur);
Map.addLayer(classe2000, {min: 1, max: 6, palette: PALETTE}, 'Classification LULC 2000');

// ===== 7. Évaluation de la précision =====
var matrice = ee.ConfusionMatrix(testSet.classify(classifieur).errorMatrix('class', 'classification'));
print('Matrice de confusion 2000 :', matrice);
print('Précision globale 2000 :', matrice.accuracy());
print('Indice Kappa 2000 :', matrice.kappa());

// ===== 8. Export GeoTIFF de la classification =====
Export.image.toDrive({
  image: classe2000,
  description: 'campo_LULC_2000',
  folder: 'campo_LULC',
  scale: 30,
  region: aoi,
  crs: 'EPSG:32632',
  maxPixels: 1e13
});

// ===== 9. Export de la carte finale en couleur =====
Export.image.toDrive({
  image: classe2000.visualize({min: 1, max: 6, palette: PALETTE}),
  description: 'campo_carte_LULC_2000',
  folder: 'campo_LULC',
  scale: 30,
  region: aoi,
  crs: 'EPSG:32632',
  maxPixels: 1e13
});
