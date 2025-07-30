from conan import ConanFile
from conan.tools.env import VirtualRunEnv, VirtualBuildEnv
from conan.tools.cmake import CMakeDeps, CMakeToolchain

required_conan_version = ">=1.53.0"


class ZetaConan(ConanFile):
    name = "zeta"
    version = "dev"
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "with_test": [True, False],
    }
    default_options = {
        "with_test": False,
    }

    @property
    def _min_cppstd(self):
        return 14

    def configure(self):
        # specifiy opencv options
        self.options["opencv"].shared = True
        self.options["opencv"].with_ffmpeg = False
        self.options["opencv"].with_tiff = False
        self.options["opencv"].with_webp = False
        self.options["opencv"].with_openexr = False
        self.options["opencv"].with_jpeg2000 = False
        self.options["opencv"].with_gtk = False
        self.options["opencv"].with_jpeg = "libjpeg-turbo"
        self.options["opencv"].text = False
        self.options["opencv"].with_cuda = True
        self.options["opencv"].cudaarithm = True
        self.options["opencv"].dnn = True
        self.options["opencv"].cudaimgproc = True
        self.options["opencv"].cudawarping = True
        self.options["opencv"].cuda_arch_bin = "7.2,7.5,8.6"
        
        # export CUDA_PATH and CUDA_LIB_PATH
        self.options["opencv"].objdetect = True
        self.options["opencv"].calib3d = True
        self.options["opencv"].imgproc = True
        self.options["opencv"].with_quirc = True

    def build_requirements(self):
        pass

    def requirements(self):
        self.requires("opencv/4.5.5@transformer/stable")
        # self.requires("opencv/4.5.5")

    def generate(self):
        tc = CMakeToolchain(self)
        tc.cache_variables["CMAKE_POLICY_DEFAULT_CMP0077"] = "NEW"
        if self.options.with_test:
            tc.variables["BUILD_TESTING"] = True
        else:
            tc.variables["BUILD_TESTING"] = False
        tc.generate()
        tc = CMakeDeps(self)
        tc.generate()
        tc = VirtualRunEnv(self)
        tc.generate()
        tc = VirtualBuildEnv(self)
        tc.generate(scope="build")