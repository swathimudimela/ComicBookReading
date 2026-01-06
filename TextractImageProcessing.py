from textractor import Textractor
from textractor.data.constants import TextractFeatures

photo = 'birbalComicPage2.jpg'

extractor = Textractor(profile_name = "default")

result = extractor.analyze_document(file_source = photo,
                                    features = [TextractFeatures.LAYOUT],
                                    save_image = True)

result.pages[0].visualize()
result.pages[0].page_layout.titles.visualize()
result.pages[0].page_layout.headers.visualize()

result.pages[0].page_layout.section_headers.visualize()
result.pages[0].page_layout.footers.visualize()
result.pages[0].page_layout.tables.visualize()
result.pages[0].page_layout.key_values.visualize()
result.pages[0].page_layout.page_numbers.visualize()
result.pages[0].page_layout.lists.visualize()
result.pages[0].page_layout.figures.visualize()