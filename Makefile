MODEL = Qwen/Qwen3.5-0.8B
CASES = $(notdir $(wildcard cases/*))
UPLOT = 1.6.32
UPLOT_DIR = static/vendor/uplot

define HELP
Targets:
  help        show this message
  download    fetch the model $(MODEL), the script itself runs offline
  gray        gray image to "This is a picture of a cat.", output in out_gray
  uplot       fetch uPlot $(UPLOT) for the viewer into $(UPLOT_DIR)
  viewer      web viewer of the runs in cases/, http://127.0.0.1:8000
  pages       clone the gh-pages branch (the report) into pages/
  check       ruff lint and format check
  fix         ruff lint with fixes and format
  clean       remove out_*
Cases, run cases/NAME/run.sh with output next to the script:
  $(CASES)
endef

help:
	$(info $(HELP))
	@:

download:
	HF_HUB_DISABLE_TELEMETRY=1 hf download $(MODEL)

gray:
	./metamers.py --model $(MODEL) --init gray --out out_gray

uplot:
	mkdir -p $(UPLOT_DIR)
	curl -sSfL https://registry.npmjs.org/uplot/-/uplot-$(UPLOT).tgz | tar xz -C $(UPLOT_DIR) --strip-components=1 \
		package/LICENSE package/dist/uPlot.iife.min.js package/dist/uPlot.min.css
	mv $(UPLOT_DIR)/dist/* $(UPLOT_DIR)/ && rmdir $(UPLOT_DIR)/dist

viewer:
	./viewer.py

$(CASES):
	cases/$@/run.sh

pages:
	git clone -b gh-pages --single-branch git@github.com:pkarnakov/metamers.git pages

check:
	ruff check
	ruff format --check

fix:
	ruff check --fix
	ruff format

clean:
	$(RM) -r out_*

.PHONY: help download gray uplot viewer check fix $(CASES) clean
